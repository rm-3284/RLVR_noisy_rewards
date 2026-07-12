"""Probe a Gemma INSTRUCT model on toy MATH: does it box reliably, terminate cleanly, AND get answers right?
Uses the chat template (instruct format) + greedy decode. Reports boxed-rate + correctness + token length."""
import os, re, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ["GEMMA_MODEL"]
print(f"MODEL={MODEL}", flush=True)
PROBS = [
    ("What is 7 * 8?", "56"),
    ("If 3x + 5 = 20, what is x?", "5"),
    ("A rectangle has length 8 and width 3. What is its area?", "24"),
    ("Evaluate the sum 1 + 2 + 3 + ... + 100.", "5050"),
    ("Find the value of x if x^2 - 5x + 6 = 0 and x > 2.", "3"),
]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
            device_map="cuda", attn_implementation="eager").eval()

nbox = ncorr = 0
for q, gold in PROBS:
    msg = [{"role": "user", "content": f"Solve the problem. Put your final answer in \\boxed{{}}.\n\nProblem: {q}"}]
    inp = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=512, do_sample=False)
    text = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
    hasbox = "\\boxed" in text
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    boxed = (m.group(1).strip() if m else "")
    corr = boxed.replace(" ", "") == gold
    ntok = out.shape[1] - inp.shape[1]
    nbox += hasbox; ncorr += corr
    print("=" * 72)
    print(f"Q: {q} | gold={gold}")
    print(f"  boxed={hasbox} extracted='{boxed}' correct={corr} ntok={ntok} terminated={ntok < 500}")
    print("  " + text[:400].replace("\n", "\n  "))
print("=" * 72)
print(f"SUMMARY: {nbox}/5 boxed, {ncorr}/5 correct")
