# RLVR under Noisy Verifiers — Research Notebook

Running, concise, step-by-step log of what was done, what was found, and what's open.
Figures in `notebook/figures/`. Detailed per-experiment logs in `../logs/*.md`. Updated iteratively.

**Setup:** GRPO (NeMo-RL) with a controlled noisy verifier. On each rollout the clean binary reward is
flipped by i.i.d. Bernoulli noise: **fp** = P(flip wrong→rewarded), **fn** = P(flip right→unrewarded)
(`nemo_rl/environments/math_environment.py`). Margin **m = 1 − fp − fn** = expected reward gap between a
correct and a wrong answer. Qwen2.5 (0.5/1.5/3B) + OLMo-2-1B; GSM8K + MATH; r (rollouts/prompt) = 4…256.
5 seeds/cell; report converged validation accuracy.

---

## Step 0 — Port & baseline
- Ported the noisy-RLVR pipeline to Della (air-gapped compute: fat prebuilt venv, offline HF/datasets, QOS routing). Disabled ~8GB/run rollout dumps (were filling scratch).
- Baseline dense 6×6 fp/fn grid on GSM8K-1.5B (r=32, 5 seeds).

## Step 1 — Margin-collapse (the first-order law)
- On GSM8K-1.5B, accuracy depends on **m alone**: all 15 matched-margin FP↔FN mirror pairs agree within ±0.03. FP and FN are **interchangeable** here. → **Fig 2 (left).**
- Holds across sizes (0.5B, 1.5B, 3B) — level shifts, form unchanged.
- **Verdict:** margin-collapse `acc = f(m)` is the validated first-order law (bounded/saturating in m).

## Step 2 — MATH & the asymmetry (collapse BREAKS)
- MATH-1.5B (levels 1-3, base rate 0.42): at matched m=0.7, **FP(0.3,0)=0.216 vs FN(0,0.3)=0.381 → −0.166, FP-worse.** Collapse breaks. → **Fig 2 (right).**
- MATH-3B (base 0.73): symmetric again (−0.005). Same task/verifier, only base rate differs.
- **First claim:** which error is worse is **base-rate-moderated**, not task-fixed.

## Step 3 — Decouple base rate from capacity
- MATH across sizes: base rate is **non-monotonic** in size (0.5B=0.59, 1.5B=0.42, 3B=0.73), yet the **only** low-base config (1.5B) is the **only** FP-worse one.
- Cleanest control: **MATH-1.5B, same model/task/verifier, only difficulty changed** — levels 1,2,3 (base 0.42) FP-worse −0.165 vs levels 1,2 (base 0.88) symmetric −0.015. Nothing varies but base rate → asymmetry flips. Rules out size, task, family, verifier.

## Step 4b — 2nd VERIFIER TYPE: code (execution/unit-test) — collapse holds, FP-worse is dynamics
- MBPP with a real execution verifier (run candidate vs hidden asserts; `code_verifier_environment.py`, subprocess+timeout+fp/fn). **Both converged code configs SYMMETRIC**: 0.5B-MBPP (base 0.40, +0.006), 1.5B-MBPP (base 0.51, −0.016). → **margin-collapse VALIDATED on a fundamentally different verifier type** (not string/equivalence) — the law isn't math-verifier-specific.
- **FP-worse did NOT reproduce on code** even at base 0.40 (where MATH-1.5B @0.42 was strongly FP-worse). Reason: code models **bootstrap up to 0.40–0.51 and ESCAPE the low-precision trap**; MATH-1.5B stays trapped at 0.42. → **FP-worse is governed by whether the model escapes the early low-precision trap (training dynamics), not converged base rate alone.** Honest refinement of the base-rate story. (logs/CODE_probe.md)

## Step 4 — Kill the Qwen confound (non-Qwen family)
- OLMo-2-1B (different family) GSM8K (base 0.58): **symmetric** → symmetric regime not a Qwen artifact.
- OLMo-2-1B **MATH** (base 0.12, low): **FP-worse −0.051 (n=5)** → the FP-worse regime **also** reproduces off-Qwen.
- **Both regimes confirmed on a second model family.**

## Step 5 — The money picture (asymmetry vs base rate)
- 9 configs (2 tasks × 3 sizes × 2 families). **FP/FN ratio is monotone in base rate**: 0.36 (b=0.12) → 0.57 (b=0.42) → ≈1.0 (b≥0.53), sharp **threshold b≈0.45–0.5**. → **Fig 1.**
- The *absolute* FP−FN is a hump (peaks ~0.42, compressed near the [0,1] floor at 0.12) — a bounding artifact; the **ratio** is the clean measure.

## Step 6 — Mechanism
- FP degrades reward **precision** (rewards truly-wrong rollouts); FN degrades **recall** (fewer correct rollouts rewarded). Learning depends on precision.
- FP-worse switches on only when precision drops below a **critical bootstrapping threshold** (early training current-acc≈0 → at high fp almost every rewarded rollout is wrong → can't climb out). Explains the *sharp* base-rate transition. (Extended-range runs show high-fp cells sit at the floor early then escape if the model bootstraps high enough — permanent deficit only when converged capability stays low.)

## Step 7 — Extended noise range (fp/fn up to 1) — phase transition
- Pushed beyond the usual fp,fn≤0.5. Accuracy has a **phase structure in m**: **learn** (m>0, collapse/symmetric even at fp/fn=0.9) → **frozen at init** (m=0, no signal) → **learn-to-be-WRONG** (m<0, driven to 0). → **Fig 4.**
- m<0 ("adversarial verifier") is a sanity/boundary result — the model faithfully optimizes any reward, even inverted. Not the paper's focus (nobody deploys a sub-random verifier); kept as the boundary of the phase diagram.

## Step 8 — Compute axis (Headline B)
- Qwen-1.5B GSM8K, r = 4…256: 0.684/0.709/0.734/0.762/0.778/0.781/**0.804**. → **Fig 3.**
- **STRONG DIMINISHING RETURNS**: per-doubling gain falls from ~0.025 (r≤32) to ~0.015 (r≥64). But the curve is **still slowly creeping up to ~0.80 at r=256** — it does NOT cleanly saturate within the measured range. Whether it's a flat plateau or a slow continued rise is **unresolvable within the ±0.02 seed oscillation** (the high-r per-doubling gains +0.003/+0.023 bounce within noise). Honest claim: heavy diminishing returns to ~0.80; no clean asymptote.
- Noise (0.2,0.2) = a **small (~0.04–0.05), roughly compute-independent offset**; noisy plateaus ~0.73. At high base rate you **can't buy back verifier noise with compute**.
- (Integrity note: I revised this twice — "gap widens" and "saturates ~0.78" both retracted as oscillation-limited; the robust claim is *diminishing returns*, not a specific asymptote.)
- **Batch-confound control (defends this):** at FIXED batch (1024) + FIXED steps (233), accuracy rises with rollouts-per-prompt (R=8/16/32/64 → 0.741/0.759/0.762/0.766) *despite high-R seeing less data* → the compute gains are from better advantage estimation (more rollouts/prompt), NOT batch size. Kills the "is it just batch scaling?" objection. (logs/C2_batch_control.md)

## Step 9 — Compute × base-rate interaction (A×B)
- Transfer to 0.5B: the noise gap is **large at low compute and shrinks with compute** (0.148 @r8 → 0.112 @r32) — **opposite** to 1.5B (gap ~constant). → **Fig 5.**
- Reading: at **low** capability compute **does** buy back noise (raising accuracy→precision); at **high** capability the penalty is already small and compute-independent. So "compute can't rescue noise" is **regime-specific**. (0.5B r=128 pending to confirm.)

## Step 10 — REFRAME: the asymmetry is PRECISION-moderated (a low-precision TRANSIENT), not converged-base-rate
- A difficulty sweep (MATH-1.5B lv1-4 / lv1-5, n=5, clean/FP/FN trajectories) exposed that the **Step 2 "lv1-3 base 0.42 FP-worse −0.166" was an UNDERTRAINED snapshot** — that run hit the **epoch trap** (max_num_epochs=1 × a small level-filtered dataset → ~20 steps), caught mid-climb. Trained longer, the *same setup* climbs to ~0.75 and **FP catches up → the asymmetry vanishes.** → **Fig 6.**
- **n=5 trajectories (both bins, identical):** at the trapped phase (step 20, precision ~0.3–0.4) **FP−FN ≈ −0.18** (reproduces the −0.166), then **monotonically → ~0 (symmetric) at plateau** (mL4 −0.007, mL5 −0.004) as precision climbs.
- **The organizing variable is the low-precision BOOTSTRAPPING phase, not converged base rate.** FP-worse is concentrated at low precision / early steep climbing (Fig 6 right); it resolves once the model settles near its ceiling. The clincher on precision-vs-time: **MATH-3B (0.73) was *also* undertrained yet symmetric** (high precision → already escaped), while **OLMo-MATH (0.11) trains 100+ steps but stays trapped → still FP-worse** (the one *converged* FP-worse anchor). So it's precision/dynamics, not step count.
- **Honest caveat (magnitude is task-dependent, NOT a universal 1D collapse):** at matched precision ~0.3–0.4, MATH is strongly FP-worse (mL4 −0.09 to −0.18) but code is barely (−0.02; settled code@0.40 = +0.006). The cleaner variable is fragile *bootstrapping dynamics* (steep climb, model depends on the reward signal) — and MATH (open-answer/string verifier) is more FP-susceptible than code (execution verifier). Robust claim = the **direction** + the **within-run trajectory** + the **OLMo permanent-trap anchor**; do NOT claim a universal quantitative precision law.
- **Why this is a refinement, not a break:** the −0.166 result was real (the trapped-phase value); only its *label* ("converged base rate 0.42") was wrong. The mechanism (Step 6) is unchanged and now has direct within-run causal evidence. Headline: **which error is worse is governed by whether the model is currently in the low-precision bootstrapping trap** — a training-dynamics effect; capability × task-difficulty × training-length set how fast it escapes.
- **Second independent line (EXT_probe, GSM8K extreme noise):** the same transient→resolve pattern appears with a *different* task and *extreme* noise: GSM8K-1.5B (high base) at **fp=0.9** has FP lagging FN mid-training (0.27 vs ~0.59 at step ~48) but converges **symmetric** (FP 0.573 ≈ FN 0.578 at n=5). So even a catastrophic false-positive rate is a *transient bootstrapping delay* when the model can climb out — permanent FP-worse needs low *converged* capability (MATH-1.5B stuck, OLMo). The escape/dynamics reframe thus rests on **two independent lines** (MATH-difficulty/moderate-noise trajectories + GSM8K/extreme-noise convergence) plus the OLMo permanent-trap anchor.
- **Methods note (epoch trap):** MATH base rates measured at max_num_epochs=1 with level-filtered datasets are training snapshots of varying length; the precision axis makes this immaterial (asymmetry tracks precision, not step count). GSM8K main results are unaffected (fixed large dataset → consistent, converged ~step 125).

---

## Functional-form status (rigor)
- **Collapse `acc=f(m)`** — validated (held-out).
- **Asymmetry** — framed as a **low-precision bootstrapping effect**: FP-worse concentrated at low precision / early steep climbing, resolving as the model settles (Fig 6). This is a **direction + dynamics** claim, NOT a universal quantitative form — the magnitude is task/verifier-dependent (MATH strongly FP-worse during its climb, code only mildly; §Step 10 caveat). The robust, defensible evidence: the within-run trajectory (mL4/mL5: FP-worse early → symmetric late, n=5), the OLMo-MATH converged permanent-trap anchor, and the cross-config direction. A clean single-variable g(precision) does **not** hold across tasks, so we frame asymmetry as a mechanistically-grounded regime effect, with margin-collapse as the validated first-order form.

## Open / owed
1. **Task-axis validation** (next major thrust, also supplies low-base configs): **GPQA** first (adapt eval loader → MC verifier + noise; mind the ~0.25 MC floor), then **MBPP/code** (execution verifier — genuinely different; build unit-test→reward on the existing code sandbox). See `../logs/task-set-plan` / memory.
2. Finalize **r=256** (pin the compute asymptote) and **0.5B r=128** (confirm A×B).
3. **7B** (capacity ceiling) — needs 4-GPU. Llama-1B base can't bootstrap GSM8K (dropped).
4. Precision/threshold **form fit** once low-base configs densified.

## Integrity ledger (leans retracted on fuller data)
"gap widens with compute", a premature "saturation" read (oscillation snapshot), an "FN-worse-at-high-base mirror" (washed out at n=5), and "asymmetry as a smooth global 2-var form" (rejected held-out). **Biggest correction (Step 10): "asymmetry moderated by *converged base rate*" → moderated by *current precision*; the "MATH-1.5B base 0.42" cornerstone was an undertrained epoch-trap snapshot, caught by an n=5 difficulty sweep whose runs escaped the trap.** The finding survived (trapped-phase FP−FN ≈ −0.18 reproduced) but the label was corrected. Nothing claimed past its evidence.
