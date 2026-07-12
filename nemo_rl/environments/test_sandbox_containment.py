"""Containment test for the bwrap code sandbox (nemo_rl/environments/code_verify.py).
Proves untrusted model code is contained BEFORE any code task is trusted. Run:
    uv run --no-sync --offline python nemo_rl/environments/test_sandbox_containment.py
Every check must PASS (all 6) or the sandbox is NOT safe and no code task may run.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nemo_rl.environments import code_verify as cv  # noqa: E402

SCRATCH_MARKER = "/scratch/gpfs/GRIFFITHS/aw2418/SANDBOX_ESCAPE_TEST.txt"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# 0. sandbox present at all (fail-closed sanity)
check("bwrap present", cv._HAVE_BWRAP, "/usr/bin/bwrap")

# 1. LEGIT code still works (sandbox must not break normal execution)
legit = cv.run_tests("def add(a,b):\n    return a+b", "", ["assert add(2,3)==5", "assert add(-1,1)==0"])
check("legit solution passes", legit is True, "correct add() scores pass")
# and a legit stdin/stdout APPS-style program
apps_ok = cv.run_apps("print(sum(int(x) for x in input().split()))",
                      {"inputs": ["2 3\n"], "outputs": ["5\n"]})
check("legit stdin program passes", apps_ok is True, "sum-of-stdin scores pass")

# 2. NO host filesystem write — the incident's exact failure (junk files in /scratch)
try:
    if os.path.exists(SCRATCH_MARKER):
        os.remove(SCRATCH_MARKER)
except Exception:
    pass
cv._run_untrusted(f"open({SCRATCH_MARKER!r}, 'w').write('ESCAPED')", timeout=8)
check("no host FS write to /scratch", not os.path.exists(SCRATCH_MARKER),
      "marker file was NOT created on host")

# 3. NO write to system dirs even if it tries (ro-bind)
r = cv._run_untrusted("open('/usr/PWNED','w').write('x'); print('WROTE_USR')", timeout=8)
check("no write to /usr (read-only)", "WROTE_USR" not in (r.stdout or "") and not os.path.exists("/usr/PWNED"),
      "system dir stayed read-only")

# 4. NO network
r = cv._run_untrusted(
    "import socket\ntry:\n socket.create_connection(('8.8.8.8',53),timeout=4); print('NET_OK')\nexcept Exception as e:\n print('NET_BLOCKED')",
    timeout=12)
check("network blocked", "NET_OK" not in (r.stdout or ""), f"stdout={ (r.stdout or '').strip()[:40]!r}")

# 5. NO privilege escalation — sudo/id must not yield real root (uid 0 outside ns)
r = cv._run_untrusted(
    "import subprocess\ntry:\n o=subprocess.run(['sudo','-n','id'],capture_output=True,text=True,timeout=5); print('SUDO_RC',o.returncode)\nexcept Exception as e:\n print('SUDO_ERR')",
    timeout=10)
check("sudo yields no root", "uid=0(root)" not in (r.stdout or "") and "SUDO_RC 0" not in (r.stdout or ""),
      f"stdout={(r.stdout or '').strip()[:50]!r}")

# 6. resource-bounded — infinite loop is killed (CPU rlimit / wall timeout), returns fail not hang
import time  # noqa: E402
t0 = time.time()
loop_ok = cv.run_tests("def add(a,b):\n    while True: pass", "", ["assert add(1,1)==2"], timeout=5)
check("infinite loop contained", loop_ok is False and (time.time() - t0) < 40,
      f"loop killed in {time.time()-t0:.1f}s, scored fail")

n_pass = sum(1 for _, ok, _ in results if ok)
print(f"\nSANDBOX CONTAINMENT: {n_pass}/{len(results)} PASS")
sys.exit(0 if n_pass == len(results) else 1)
