"""Observe gemma-2-2b raw MATH generations to diagnose format-marginality (does it emit \\boxed{}?).
transformers + eager attention (handles gemma-2 tanh softcapping natively, avoids the vLLM/FlashInfer setup)."""
import os, torch
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL=os.environ.get("GEMMA_MODEL","/scratch/gpfs/GRIFFITHS/aw2418/hf_models/gemma-2-2b")
print(f"MODEL={MODEL}", flush=True)
SYS = open("examples/prompts/math.txt").read().strip()
PROBS = [
    "What is 7 * 8?",
    "If 3x + 5 = 20, what is x?",
    "A rectangle has length 8 and width 3. What is its area?",
    "Evaluate the sum 1 + 2 + 3 + ... + 100.",
    "Find the value of x if x^2 - 5x + 6 = 0 and x > 2.",
]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
            device_map="cuda", attn_implementation="eager")
model.eval()
torch.manual_seed(0)
nbox=0
for p in PROBS:
    prompt = f"{SYS}\n\nProblem: {p}\nSolution:"
    inp = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=512, do_sample=True, temperature=1.0, top_p=1.0)
    text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    has_box = "\\boxed" in text
    nbox += has_box
    print("="*80); print("PROBLEM:", p)
    print("HAS_\\boxed{}:", has_box, "| new_tokens:", out.shape[1]-inp.input_ids.shape[1])
    print("OUTPUT >>>"); print(text[:1400])
print("="*80); print(f"SUMMARY: {nbox}/{len(PROBS)} outputs contained \\boxed{{}}")
