# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Core code-verification for MBPP-style tasks: run candidate against hidden asserts.

Untrusted model-generated code → executed in a SUBPROCESS with a TIMEOUT and a
reliability guard (disable file writes / os.system / network) — the standard bigcode-eval
safety pattern. Pure functions here (no env deps) so they can be unit-tested standalone.
reward = 1.0 if the candidate passes ALL asserts, else 0.0. (fp/fn noise added by the env.)
"""
import json
import os
import re
import resource
import subprocess
import sys

# Preamble injected before untrusted code: neutralizes the obvious dangerous calls + limits CPU.
_GUARD = """
import os as _os, sys as _sys, signal as _signal, resource as _resource
try:
    _resource.setrlimit(_resource.RLIMIT_CPU, (8, 8))
    _resource.setrlimit(_resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
except Exception: pass
for _n in ("system","popen","remove","unlink","rmdir","rename","kill"):
    try: setattr(_os, _n, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("blocked")))
    except Exception: pass
__import__("builtins").open = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("io blocked")) \
    if (len(a) > 1 and any(m in str(a[1]) for m in ("w","a","x","+"))) else __import__("io").StringIO("")
"""

# ---- OS-level sandbox (bubblewrap) — the REAL barrier; _GUARD above is only defense-in-depth. ----
# bwrap gives an isolated user + network + mount namespace: no network, no privilege escalation
# (sudo/setuid structurally impossible in the user ns), and NO host filesystem except read-only
# /usr + an ephemeral tmpfs /tmp — so untrusted code cannot reach /scratch|/home or the network.
# Verified by nemo_rl/environments/test_sandbox_containment.py (run before trusting any code task).
_HAVE_BWRAP = os.path.exists("/usr/bin/bwrap")
_BWRAP = [
    "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
    "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp",
    "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
    "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
    "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
    "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin",
    "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/tmp",
]
_SANDBOX_PY = "/usr/bin/python3"
# Minimal env for the sandbox process (bwrap 0.4.1 has no --clearenv; controlling env here is equivalent).
_SANDBOX_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}

# ---- Apptainer sandbox (PREFERRED) — bwrap needs user namespaces, which Della disabled
# (max_user_namespaces=0) in an Aug-2026 security update. Apptainer runs setuid (privileged starter),
# so it builds its namespaces as root and works even with userns off. --containall = isolated fs (NO host
# binds -> /scratch,/home unreachable) + clean env + pid/ipc ns; --net --network none = no network;
# --writable-tmpfs = ephemeral overlay; --memory/--pids-limit = cgroup caps (memory + fork bombs).
# Untrusted code runs as the unprivileged user inside -> no privilege escalation.
# Proven by nemo_rl/environments/test_sandbox_containment.py (all 6 checks must pass).
_APPTAINER_SIF = os.environ.get("CODE_SANDBOX_SIF", "/scratch/gpfs/GRIFFITHS/aw2418/code_sandbox.sif")
_HAVE_APPTAINER = os.path.exists("/usr/bin/apptainer") and os.path.exists(_APPTAINER_SIF)
_APPTAINER = [
    "apptainer", "exec", "--containall", "--net", "--network", "none",
    "--no-home", "--pwd", "/tmp",
    # NO --memory/--pids-limit: they need a user cgroup (XDG_RUNTIME_DIR) that a Slurm job doesn't provide
    #   -> apptainer FATALs inside a job. Resource bounds come instead from Slurm's own job cgroup (--mem
    #   OOM-kills the job) + the wall timeout + the PID namespace (--containall). NO --writable-tmpfs:
    #   rootfs (incl /usr) stays READ-ONLY; --containall gives a writable isolated tmpfs /tmp.
    _APPTAINER_SIF,
]


def _sandbox_rlimits():
    """Resource caps inherited into the sandboxed child: CPU, address space, file size.
    NB: do NOT set RLIMIT_NPROC — it counts processes per real-uid GLOBALLY, so with the user's
    many concurrent jobs bwrap itself can't fork ('namespace creation: Resource temporarily
    unavailable'). Fork bombs are instead bounded by the PID namespace (--unshare-all) + the wall
    timeout + --die-with-parent teardown, plus RLIMIT_CPU/AS per process."""
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    lim = getattr(resource, "RLIMIT_FSIZE", None)
    if lim is not None:
        try:
            resource.setrlimit(lim, (16 * 1024**2, 16 * 1024**2))
        except Exception:
            pass


def _run_untrusted(program: str, timeout: float, stdin: str | None = None):
    """Execute untrusted `program` in an isolated sandbox. FAIL CLOSED: refuse if none available.
    Prefers Apptainer (works where user namespaces are disabled — Della); falls back to bwrap (Ionic)."""
    if _HAVE_APPTAINER:
        # --containall cleans the env inside the container; the outer apptainer inherits env to find itself.
        return subprocess.run(
            _APPTAINER + ["python3", "-I", "-c", program],
            input=stdin, capture_output=True, timeout=timeout, text=True,
        )
    if _HAVE_BWRAP:
        return subprocess.run(
            _BWRAP + [_SANDBOX_PY, "-I", "-c", program],
            input=stdin, capture_output=True, timeout=timeout, text=True,
            preexec_fn=_sandbox_rlimits, env=_SANDBOX_ENV,
        )
    raise RuntimeError(
        "SANDBOX UNAVAILABLE: neither Apptainer+SIF nor bwrap — refusing to execute untrusted code")


def extract_code(text: str) -> str:
    """Pull the last ```python ...``` block; fall back to the whole text."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1]
    return text


def run_tests(candidate: str, setup: str, tests: list[str], timeout: float = 8.0) -> bool:
    """True iff candidate + setup makes every assert in `tests` pass (in an isolated subprocess)."""
    if not candidate.strip() or not tests:
        return False
    program = _GUARD + "\n" + (setup or "") + "\n" + candidate + "\n" + "\n".join(tests) + "\n"
    try:
        return _run_untrusted(program, timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def _norm(s) -> str:
    """Normalize program output for comparison: rstrip each line, drop trailing blank lines."""
    return "\n".join(ln.rstrip() for ln in str(s).rstrip().splitlines())


def _eq_output(got: str, exp: str) -> bool:
    """Line-normalized equality, with a numeric-tolerant fallback (handles float/int formatting)."""
    if _norm(got) == _norm(exp):
        return True
    gt, et = _norm(got).split(), _norm(exp).split()
    if len(gt) != len(et):
        return False
    try:
        return all(abs(float(a) - float(b)) <= 1e-4 for a, b in zip(gt, et))
    except ValueError:
        return False


# Competitive/call-based solutions assume these are in scope (esp. List[int] type hints).
_IMPORTS = "from typing import *\nimport sys, math, collections, itertools, bisect, heapq, re, functools, string\n"

_APPS_CALL_HARNESS = """
import sys as _s
_fn_name, _inputs, _outputs = {fn!r}, {inp!r}, {out!r}
try:
    _fn = getattr(Solution(), _fn_name) if "Solution" in dir() else globals()[_fn_name]
except Exception:
    _s.exit(1)
_ok = True
for _args, _exp in zip(_inputs, _outputs):
    try:
        _r = _fn(*_args)
    except Exception:
        _ok = False; break
    _e = _exp[0] if (isinstance(_exp, list) and len(_exp) == 1) else _exp
    if _r != _e and str(_r) != str(_e):
        _ok = False; break
_s.exit(0 if _ok else 1)
"""


def run_apps(candidate: str, io: dict, timeout: float = 8.0) -> bool:
    """True iff candidate passes ALL of APPS `io` cases. Handles stdin/stdout AND call-based (fn_name)."""
    inputs, outputs = io.get("inputs") or [], io.get("outputs") or []
    if not candidate.strip() or not inputs:
        return False
    fn_name = io.get("fn_name")
    if fn_name:  # call-based: candidate defines fn_name (bare or as Solution method)
        prog = _GUARD + _IMPORTS + candidate + "\n" + _APPS_CALL_HARNESS.format(
            fn=fn_name, inp=inputs, out=outputs)
        try:
            return _run_untrusted(prog, timeout).returncode == 0
        except Exception:
            return False
    # stdin/stdout: candidate is a full program; feed each input, compare stdout
    prog = _GUARD + _IMPORTS + candidate
    for inp, exp in zip(inputs, outputs):
        inp_s = inp if isinstance(inp, str) else "\n".join(map(str, inp))
        exp_s = exp if isinstance(exp, str) else (
            "\n".join(map(str, exp)) if isinstance(exp, list) else str(exp))
        try:
            r = _run_untrusted(prog, timeout, stdin=inp_s)
        except Exception:
            return False
        if r.returncode != 0 or not _eq_output(r.stdout, exp_s):
            return False
    return True


def run_kodcode(candidate: str, test: str, timeout: float = 8.0) -> bool:
    """True iff candidate passes ALL KodCode pytest-style tests (plain-assert only).
    KodCode tests are `from solution import fn` + `def test_x(): assert fn(...)==...`. The candidate
    is expected to define `fn` (the required name is put in the prompt), so we inline the candidate,
    strip the `from solution import` line, then call every `test_*` function; any exception
    (Assertion/NameError/...) counts as a failure. Pass = at least one test ran and none failed.
    No pytest dependency — the loader filters out capsys/pytest/heavy-lib tests."""
    if not candidate.strip() or not test.strip():
        return False
    test_body = "\n".join(
        ln for ln in test.splitlines()
        if not (ln.lstrip().startswith("from solution import") or ln.lstrip().startswith("import solution"))
    )
    runner = (
        "\nimport sys as _s\n_fail = 0\n_ran = 0\n"
        "for _n, _f in list(globals().items()):\n"
        "    if _n.startswith('test_') and callable(_f):\n"
        "        _ran += 1\n"
        "        try:\n            _f()\n        except Exception:\n            _fail += 1\n"
        "_s.exit(0 if (_ran > 0 and _fail == 0) else 1)\n"
    )
    program = _GUARD + _IMPORTS + candidate + "\n" + test_body + runner
    try:
        return _run_untrusted(program, timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def score_one(prediction: str, ground_truth_json: str, timeout: float = 8.0) -> float:
    """1.0 if the prediction passes all tests, else 0.0.
    Auto-detects gt format: KodCode {pytest_test} vs MBPP {tests,setup} vs APPS {inputs,outputs[,fn_name]}."""
    try:
        gt = json.loads(ground_truth_json)
    except Exception:
        return 0.0
    code = extract_code(prediction)
    if "pytest_test" in gt:  # KodCode pytest-style (plain asserts)
        return 1.0 if run_kodcode(code, gt["pytest_test"], timeout) else 0.0
    if "inputs" in gt:  # APPS io-based
        return 1.0 if run_apps(code, gt, timeout) else 0.0
    return 1.0 if run_tests(code, gt.get("setup", ""), gt.get("tests", []), timeout) else 0.0


if __name__ == "__main__":
    # ---- standalone self-test ----
    gt = json.dumps({"tests": ["assert add(2,3)==5", "assert add(-1,1)==0"], "setup": ""})
    good = "```python\ndef add(a,b):\n    return a+b\n```"
    wrong = "```python\ndef add(a,b):\n    return a-b\n```"
    empty = "I don't know."
    loop = "```python\ndef add(a,b):\n    while True: pass\n```"
    print("correct  ->", score_one(good, gt), "(expect 1.0)")
    print("wrong    ->", score_one(wrong, gt), "(expect 0.0)")
    print("empty    ->", score_one(empty, gt), "(expect 0.0)")
    print("infloop  ->", score_one(loop, gt, timeout=3), "(expect 0.0 via timeout)")
