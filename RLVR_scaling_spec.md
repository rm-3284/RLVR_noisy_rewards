# Handoff Spec — Compute–Supervision Scaling Laws in RLVR

**Status:** upgrade of the CTB@ICML 2026 workshop paper into a scaling-law-genre contribution.
**Repo:** `rm-3284/RLVR_noisy_rewards` (NeMo-RL + configurable noisy verifier).
**Audience:** a coding agent (Claude Code) executing against the repo, plus the authors reading along.

---

## 0. Read this first — what genre we are now in, and why it raises the bar

The workshop paper is a *directional* result ("compute doesn't fully substitute for verifier quality; FN worse than FP"). We are converting it into a **scaling-law paper**: the deliverable is the *quantified relationship* — a calibrated surface over (FP rate `x`, FN rate `p`, compute `r`, model size `M`) plus two findings that make the surface a result rather than a measurement:

- **Headline A (the surprise):** the FP/FN asymmetry, and whether it is real, mechanistic, and *regime-dependent*.
- **Headline B (the number):** the iso-accuracy **compute↔verifier exchange rate** — how much extra compute buys back a unit of noise, and whether that cost diverges (i.e. compute *cannot* fully substitute).

The genre is unforgiving in three ways the workshop version is not. These are **entry requirements**, not nice-to-haves:

1. **Multiple seeds everywhere.** A law is a curve fit; a fit through one-seed points is a sketch. Every reported coefficient/asymptote needs a CI.
2. **A functional form defensible as a law** — either mechanistically motivated (the margin/SNR collapse below) or validated by **held-out extrapolation** (CV, not same-grid fit/eval).
3. **A sweep range wide enough to pin the shape.** `log r` over `{8,16,32}` cannot distinguish log from saturating from power — three points constrain nothing. Compute must be swept wide enough to *test saturation*, because saturation-vs-not **is** the central scientific question.

If a step trades rigor for breadth, stop and flag it.

### Internal tensions in the current fit (motivation for the reframe)
The current surface `y ≈ a x² + b xp + c p² + d x + e p + f log r + g` has problems a reviewer will find:
- The **linear FN coefficient `e` is positive** (1.5B Final: `+0.577 p`) while the paper claims "FN monotonically hurts." The marginal vertex in `p` (at `x=0`) sits near `p≈0.27`, not 0; the `p=0` optimum only appears after the large negative cross-term `−1.978 xp` is folded in. The headline claim is an artifact of *joint* extremization, not a clean marginal fact.
- The **FP linear coefficient flips sign with model size** (`+0.565` at 1.5B, `−0.088` at 0.5B). A naive base-rate story predicts the *opposite* ordering. Either there's real capacity-gating to explain, or it's seed noise. **We do not currently know which, and the entire reconciliation narrative depends on the answer.** → Phase 0 Gate.
- The polynomial is a Taylor expansion around `(0,0)` extremized over all of `[0,0.5]²`, and is fit+evaluated on the same grid with no CV.

The reframe (Phase 2) replaces 7 fitted coefficients with **one derived axis**, which is what makes it law-shaped rather than curve-fit-shaped.

---

## 1. The scientific objects

### 1.1 Dependent variable
**True validation accuracy**, evaluated with the *exact* task verifier (noise applied only during training; see §1.6 for the task set and the special handling math requires). Track **both**:
- `acc_final` — last-checkpoint (main text);
- `acc_best` — best-across-training (robustness);
- **and `acc(t)` over training steps** — because "training *reward* plateaus" (the one-epoch justification) is a plateau in the *noisy* reward, which is **not** evidence of a plateau in true accuracy. Log true val accuracy on a fixed eval interval so under-convergence can be ruled out as a source of the "gap."

### 1.2 Scaling axes
| Axis | Symbol | Core range | Notes |
|---|---|---|---|
| False-positive rate | `x` | `{0,0.1,0.2,0.3,0.4,0.5}` | flip `y*=0 → 1` w.p. `x` |
| False-negative rate | `p` | `{0,0.1,0.2,0.3,0.4,0.5}` | flip `y*=1 → 0` w.p. `p` |
| Compute (rollouts/prompt) | `r` | `{4,8,16,32,64,128}` (+`256` if budget) | wide enough to test saturation |
| Model size | `M` | `{0.5B, 1.5B, 3B, 7B}` Qwen2.5 | **now a first-class axis** (see §1.5) |

Batch size fixed at 32 (preliminary work showed negligible effect; keep, but log it).

### 1.3 The compute axis must be able to saturate (the asymptote test)
For each `(p,x,M)`, fit a **saturating** curve in `r`, not `log r`:

```
acc(r) = A − B · r^(−c)          # power-law approach to asymptote A   (A,B,c > 0)
# alternative form, fit both, compare by held-out / AIC:
acc(r) = A · (1 − exp(−r/τ))     # exponential saturation
```

Estimate the **asymptote `A_{p,x,M}` with a bootstrap CI** (resample seeds → refit → distribution of `A`). The central question becomes a clean hypothesis test:

> **H0: `A_{p,x} = A_{0,0}`** (compute fully closes the noise gap → Rad et al.'s "fate is fine, only rate differs" holds in practice).
> **H1: `A_{p,x} < A_{0,0}`** (gap persists asymptotically → theory's asymptotic claim does not transfer at realistic budgets).

The **sign and CI of `A_{p,x} − A_{0,0}` *is* the paper's central result.** Note this directly engages your Appendix C observation (`(p,r)=(0.3,64)` beats `(0,4)`), which is qualitatively in tension with the "gap persists" main-text framing — the asymptote test *resolves* that tension quantitatively.

### 1.4 The margin / SNR collapse (load-bearing — this is what makes it a law)

In GRPO the advantage is `(r_i − μ)/σ` over a group; what drives learning is the **separation between correct and incorrect rollouts**. Under noise:

- expected reward of a **correct** rollout: `1 − p`
- expected reward of an **incorrect** rollout: `x`
- **expected signal margin:** `m = (1 − p) − x = 1 − p − x`

**`m` is symmetric in `p` and `x`.** This is the whole game:

- If accuracy collapses onto `m` (or an SNR built from it), **there is no fundamental FP/FN asymmetry** — the apparent asymmetry was a signal-strength confound.
- **Any residual asymmetry that survives after conditioning on `m` is a real, isolated effect** — exactly the thing worth reporting, cleanly separated from signal strength.

Candidate predictors to test for collapse (don't pre-commit to one; test which the data collapse onto):
```
P1: m                                   # margin alone
P2: m / σ_group                          # margin over realized within-group reward std
P3: m · sqrt(ρ(1−ρ)) / sqrt(σ_flip²)     # SNR;  σ_flip² = ρ·p(1−p) + (1−ρ)·x(1−x)
```
where `ρ` = measured base pass rate of model `M` on the task. (Group composition — fraction all-correct / all-incorrect / mixed — is logged in Phase 0 so `P2`/`P3` can be computed and the tie-breaking mechanism checked directly: noise breaks all-correct/all-incorrect ties that otherwise contribute ~zero gradient.)

**Deliverable of §1.4:** a single scatter of `acc` vs the best predictor across *all* `(p,x,r,M,seed)`. Tight collapse → you've replaced the surface with one mechanistic axis. Visible residual structured by `(p−x)` → that residual is Headline A.

### 1.5 Model size is constitutive, not garnish
A law fit only at 0.5B/1.5B cannot say whether its coefficients are **stable** or **themselves scale with capacity** — and your own sign-flip says they might not be stable. Adding 3B/7B answers "do we have *a* law or *the* law," and tests whether the FP-helps effect is **capacity-gated** (a model must be good enough to exploit occasional spurious signal constructively). This is the **highest-value use of new compute**, above denser noise grids.

### 1.6 Tasks — and the special handling math requires
Task is a moderator axis (it controls answer-multiplicity, §4), not a free dial. Three tasks, each with a defined job:

| Task | Role | Answer multiplicity | Verifier |
|---|---|---|---|
| **GSM8K** | primary testbed (workshop continuity) | **unique** answer → FN = signal destruction | exact arithmetic checker (clean baseline) |
| **MATH** (or MATH-500) | (a) unique-answer **robustness**, (b) **natural systematic-FN instrument** | **unique** answer | string-match (noisy, systematic FN) **vs** Math-Verify / LLM-judge (clean reference) |
| **MBPP** | multi-solution contrast + literal Plesner comparison (§4b) | **many** correct solutions → FN ≈ forced exploration | unit tests (clean) + model-based verifier (natural systematic FP) |

**Why math is in, despite Plesner dropping it.** Plesner excluded math for two reasons that pull in *opposite* directions for us:

1. **Verifier-equivalence failure is a feature, not a bug.** A string-match verifier consistently rejects mathematically-equivalent forms (`1/e² − 1/6` vs `(6−e²)/6e²`). That is a **naturally occurring, systematic, non-i.i.d. false negative** — the exact thing the Phase 4 exploitability axis injects synthetically. Math gives it to us for free, on a real verifier, which is *more* honest than a synthetic flip. (This is the TinyV setting; Xu et al. built a method around reducing precisely these FNs.)
2. **The Qwen-math prior is genuine poison — but only single-family.** Shao et al.'s spurious-rewards result: Qwen models improve on math under *random/incorrect* reward, because RLVR partly *elicits* a strong pretraining prior rather than teaching. This is poison for our DV — if the model climbs on garbage reward, noise doesn't bite, and we'd measure an artificially flat, artificially noise-robust surface. **Worse: "FP helps" and "Qwen learns from spurious rewards" may be the same artifact wearing two hats.** So a Qwen-math number is uninterpretable *in isolation*.

**⚠ HARD RULE — never let a Qwen-math number stand alone as evidence about noise.** Always pair it so "property of noise" is separable from "property of Qwen's math prior":
- math on **Qwen + a non-Qwen family** (Llama-3.x / OLMo) on the *same* math task, **or**
- the same model on **math + a non-math task**.

Math *single-family* is the one configuration that is actively misleading; math *as a contrast* (§4c) is among the most informative things we can run. Treat math levels with suspicion; trust math *shape replication* and *cross-family divergence*.

---

## 2. Experimental design — do NOT fully cross everything

Fully crossing `6×6 noise × 6 r × 4 M × 5 seeds = 4320` runs is wasteful and is itself a tell of un-designed sweeping. Use a **fractional factorial**: dense where the science needs density, sparse elsewhere.

**CORE (non-negotiable):**
| Block | Purpose | Grid | Runs |
|---|---|---|---|
| **C1 Dense surface** | the 2D noise surface + asymmetry | `6×6 (p,x)` × `M=1.5B` × `r=32` × **5 seeds** | 180 |
| **C2 Compute sweep** | asymptote test (§1.3) | reduced noise slice† × `r∈{4,8,16,32,64,128}` × `M∈{0.5B,1.5B}` × **5 seeds** | ~480 |
| **C3 Size axis** | does the law scale (§1.5) | corner+center noise‡ × `M∈{0.5B,1.5B,3B,7B}` × `r=32` × **3 seeds** | ~120 |
| **C4 Task robustness** | does the *shape* replicate off GSM8K (§1.6) | corner+center noise‡ × **MATH** × `M∈{1.5B,7B}` × `r=32` × **3 seeds** | ~30 |

† reduced slice = diagonal `{(t,t)}` for `t∈{0,0.1,0.2,0.3}` **plus** asymmetric anchors `{(0.3,0),(0,0.3),(0.4,0),(0,0.4)}` — enough to fit `A` on both symmetric and asymmetric directions.
‡ corner+center = `{(0,0),(0.3,0),(0,0.3),(0.3,0.3),(0.15,0.15)}`.

**Task axis note:** GSM8K is the primary testbed; **MATH** enters CORE only as a *shape-replication* check (C4) — read its **shape**, not its **levels** (§1.6 hard rule). The two high-value math *experiments* (natural systematic-FN instrument, Qwen-prior confound probe) live in Phase 4, and **MBPP** enters there as the multi-solution / Plesner comparison. Any math run with a Qwen policy in C4 must have a non-Qwen counterpart before its numbers are reported (defer to §4c if 7B-Llama is unavailable at C4 time).

**EXTENDED (if budget allows, in priority order):**
1. **Moderators / Plesner reconciliation (Phase 4)** — highest scientific value of the extended set.
2. Full `6×6` surface at a second model size (test surface stability directly).
3. `r=256` arm on C2 (tighter asymptote).
4. One non-Qwen family (Llama-3.x 8B) at corner+center, for family robustness.

Tier-1 of Extended (the moderators) is arguably worth more than Extended-2/3; if compute is the constraint, prefer reconciliation over grid density.

---

## 3. Phase 0 — Infra, logging hooks, and the GATE

### 3.1 Logging hooks (add to the NeMo-RL training loop)
Per run, persist to a tidy results table (one row per eval step, plus a run-level meta row):
- run meta: `(p, x, r, M, seed, batch_size, git_sha, noise_structure)`;
- **base pass rate `ρ`** of `M` on the task (eval the *untrained* policy on train prompts; also log running `ρ` over training);
- per-step: `true_val_acc` (exact verifier), `train_reward` (noisy), `train_reward_clean` (exact, for reference);
- **group composition** per step: fraction of groups that are all-correct / all-incorrect / mixed (under *clean* labels and under *noisy* labels — both);
- realized within-group reward std `σ_group` (for predictor P2/P3).

### 3.2 Reproduce
Re-run the existing `(p,x,r)` grid from the repo at the current seed; confirm you reproduce the published numbers within run-to-run noise. Fix the environment (pin NeMo-RL, vLLM, Qwen2.5 checkpoints) and record `git_sha` per run.

### 3.3 ⛔ GATE — is the asymmetry real?
Run **C1** (the only thing gated; 180 runs, cheap at ≤1.5B). Then check, **before spending C2/C3 compute**:

- **G1 (raw, not fitted):** in the per-seed *training curves and final accuracies*, is FP-helps / FN-hurts visible across **multiple** `(p,x)` configs, or carried by 1–2? Compute the FP and FN marginal effects per seed with bootstrap CIs (paired across seeds at matched configs). Specifically test `acc(0,0.3) > acc(0,0)` and `acc(0.3,0) < acc(0,0)` as paired tests.
- **G2:** does the `0.5B vs 1.5B` FP-coefficient sign-flip survive 5 seeds at both sizes? (Needs the C3 0.5B/1.5B corner runs — pull those forward.)

**Branch:**
- **Asymmetry REAL (CIs exclude 0, multiple configs):** proceed full plan; **Headline A = reconciliation** ("which error direction is worse is regime-dependent; we reconcile Plesner-precision vs TinyV-recall as two regimes"). This is the strong paper.
- **Asymmetry is a one-seed ARTIFACT (CIs include 0):** *do not* build the reconciliation narrative. Pivot to **Headline A′ = "accuracy is governed by a single signal margin `m=1−p−x`; FP and FN are interchangeable at matched margin, contradicting prior directional claims on both sides."** This is *also* a clean, publishable scaling-law result — arguably tidier — and the margin collapse (§1.4) becomes the whole contribution. Either branch is fine; the gate just tells you which paper you're writing.

---

## 4. Phases (each: goal · configs · method · success criterion)

### Phase 1 — Seeds + the asymptote test *(genre-entry cost)*
- **Goal:** every claim gets a CI; answer "does the gap close."
- **Configs:** C1 (done in gate) + **C2**.
- **Method:** §1.3 saturating fits per `(p,x,M)`; bootstrap `A_{p,x,M}` over seeds; test `A_{p,x} − A_{0,0}`.
- **Deliverable plot:** *estimated asymptote vs noise level, with error bars*, per model.
- **Success criterion:** for each noisy config, a CI on `A_{p,x} − A_{0,0}` whose sign is determined (either direction is a result). Saturating form beats `log r` on held-out `r` (fit on `{4..64}`, predict `128`).

### Phase 2 — The law: margin collapse + CV'd surface
- **Goal:** replace the polynomial with one mechanistic axis; isolate any real asymmetry.
- **Configs:** all CORE.
- **Method:**
  1. Compute predictors P1–P3 (§1.4) per run; test collapse (R² of `acc ~ predictor`, pooled across `M,r`).
  2. Regress residuals on `(p−x)` to quantify the *isolated* asymmetry.
  3. **Empirical backstop:** fit a cross-validated surface (GP, `sklearn.gaussian_process`, RBF + white kernel; or thin-plate spline) over `(p,x,log r)`; report coefficient/length-scale CIs; **held-out** R² via k-fold over *configs* (not points) and via *leave-one-`r`-out extrapolation**.
- **Success criterion:** report the collapse R² for the best predictor and the residual-vs-`(p−x)` slope with CI. CV R² (held-out) reported alongside in-sample. The polynomial, if kept, is shown only as a local approximation with its tensions (§0) noted.

```python
# --- saturating fit + bootstrap asymptote (Phase 1) ---
import numpy as np
from scipy.optimize import curve_fit

def sat_pow(r, A, B, c):            # acc(r) = A - B r^(-c)
    return A - B * np.power(r, -c)

def fit_asymptote(r, acc_by_seed, n_boot=2000):
    """r: (n_r,), acc_by_seed: (n_seed, n_r). Returns A_hat and 95% CI."""
    A_boot = []
    seeds = np.arange(acc_by_seed.shape[0])
    for _ in range(n_boot):
        s = np.random.choice(seeds, size=len(seeds), replace=True)
        y = acc_by_seed[s].mean(0)
        try:
            popt, _ = curve_fit(sat_pow, r, y, p0=[y.max(), 0.5, 0.5],
                                bounds=([0,0,0],[1,5,5]), maxfev=10000)
            A_boot.append(popt[0])
        except RuntimeError:
            continue
    A_boot = np.array(A_boot)
    return A_boot.mean(), np.percentile(A_boot, [2.5, 97.5])

# test gap: distribution of A_{p,x} - A_{0,0}
def asymptote_gap(A_boot_noisy, A_boot_clean):
    d = A_boot_noisy[:, None] - A_boot_clean[None, :]   # all pairings
    return d.mean(), np.percentile(d, [2.5, 97.5])      # CI excludes 0 ⇒ gap real
```

```python
# --- margin collapse test (Phase 2) ---
def margin(p, x):            # symmetric signal margin
    return 1.0 - p - x
def snr_P3(p, x, rho):
    sig = margin(p, x) * np.sqrt(rho*(1-rho))
    flip_var = rho*p*(1-p) + (1-rho)*x*(1-x)
    return sig / np.sqrt(flip_var + 1e-8)
# collapse: r2_score(acc, f(predictor)); residual asymmetry: regress (acc - f(P)) on (p - x)
```

### Phase 3 — Model size as a scaling axis
- **Goal:** do coefficients/asymptotes move with capacity? Is FP-helps capacity-gated?
- **Configs:** **C3** (+ Extended-2/4 if available).
- **Method:** refit Phase-1 asymptotes and Phase-2 collapse per `M`; plot `A_{0,0}(M)`, the FP/FN marginal effects, and the residual-asymmetry slope **as functions of `M`**. Test whether the 0.5B→1.5B sign-flip continues, reverses, or stabilizes by 3B/7B.
- **Success criterion:** a "coefficients-vs-capacity" figure with CIs; a stated verdict on capacity-gating of the asymmetry.

### Phase 4 — Moderators / the Plesner reconciliation *(Headline A's payoff; Extended Tier-1)*
Two careful empirical papers (this one + Plesner) reach **opposite** prescriptions (you/TinyV: reduce FN / recall-favoring; Plesner: increase precision / FP-favoring). The reconciliation is that "which is worse" is governed by **three crossed axes**:

| Axis | This work (so far) | Plesner | Test |
|---|---|---|---|
| **Noise exploitability** | i.i.d. per-rollout flips (unexploitable) | model-based verifier = *systematic* FP (reward-hackable) | **i.i.d. FP vs systematic FP** |
| **Answer multiplicity** | GSM8K = unique answer (FN = signal destruction) | code/MBPP = many correct solutions (FN ≈ forced exploration) | **unique-answer vs multi-solution task** |
| **Base rate / capacity** | 0.5–1.5B, low `ρ` | 8–9B, `ρ≈0.85–0.90` | covered by Phase 3 |

*Exploitability is probed twice (§4a): synthetic systematic-FP on GSM8K, and natural systematic-FN on MATH via string-match-vs-equivalence verifier. Task also carries a Qwen-prior confound (§4c) that must be controlled before any math number is reported.*

- **4a Exploitability (the cleanest single axis) — probe it two complementary ways:**
  - **4a(i) Synthetic systematic-FP (controlled):** add a noise mode that flips `0→1` *deterministically* as a function of an answer feature (wrong-but-round-number, or a fixed hash bucket of the answer), so the *same* wrong output is consistently rewarded. Compare against i.i.d. FP at matched marginal rate. **Prediction:** systematic FP turns harmful (model exploits the consistent spurious signal → reward hacking, ties to Gao/Baker/Von Arx) while i.i.d. FP stays benign/regularizing.
  - **4a(ii) Natural systematic-FN on math (real verifier):** run **MATH** with a plain **string-match** verifier (whose equivalence-blindness produces *naturally* systematic false negatives) and compare against a strong **equivalence-aware reference** (Math-Verify or an LLM judge) as the "clean" verifier — the same 4B-vs-30B logic Plesner used for FP, but for FN. Set this *natural* systematic-FN against *synthetic i.i.d.* FN on the same task at matched rate. **Prediction:** systematic FN hurts more than i.i.d. FN at matched rate — a real-world demonstration of the axis that directly engages TinyV's "fix false negatives" claim. *(Subject to the §4c confound guard — pair with a non-Qwen family.)*
  - Together 4a(i)+(ii) show the exploitability axis from both error directions, one synthetic + controlled, one natural + real.
- **4b Answer multiplicity:** hold noise i.i.d., move from **unique-answer** (GSM8K, MATH) to **multi-solution** (**MBPP**, ideally Plesner's exact setup so it's a literal comparison point) and test whether "FN-worse" flips toward "FP-worse." Re-run the margin collapse per task; the residual asymmetry should track multiplicity. (Note: GSM8K↔MATH is *not* the multiplicity contrast — both are unique-answer; that pair is the robustness/Qwen-prior axis. MBPP is the multiplicity contrast.)
- **4c Qwen-prior confound probe (potential high-ceiling finding):** run **math on Qwen *and* a non-Qwen family** (Llama-3.x or OLMo, no suspiciously strong math prior) at corner+center noise. **If the noise-robustness / FP-helps effect is large on Qwen-math and *vanishes* on non-Qwen-math, that is a finding, not a failure:** it shows "FP helps" was Qwen-prior *elicitation* (Shao et al. spurious-rewards), not a property of noise — i.e. a chunk of the verifier-noise-robustness literature is downstream of pretraining, not RL dynamics. This is the deliberate trap the §1.6 hard rule sets; it converts the Qwen-math liability into a debunking-shaped result.
- **Adopt Plesner's instruments** (they're good): the **model-based-verifier-with-ground-truth** trick (realistic correlated noise *for free* while retaining exact labels for online precision/recall) and their **4-mode noise taxonomy** (sample×test, sample×rollout, group×test, group×rollout). Use the taxonomy to place your i.i.d.-vs-systematic axis precisely.
- **Positioning:** Plesner's own Limitations explicitly call for "a systematic controlled study varying FPR and FNR independently." **That is literally this paper.** Their controlled noise is *symmetric* (FPR=FNR locked), so it structurally *cannot* speak to the asymmetry; their precision claim rests entirely on **single-seed** model-based-verifier runs with precision confounded against overall verifier quality. Frame this work as running the independent-error study they flagged as missing, and finding the answer is **regime-dependent**.
- **Success criterion:** a reconciliation figure/table mapping (exploitability × multiplicity × base-rate) → which error direction dominates, that contains *both* prior findings as special cases.

### Phase 5 — Deliverables & stats hygiene
- **The exchange-rate plot (Headline B):** draw **iso-accuracy contours** in `(compute r, noise)` space from the fitted surface; report the **compute multiplier to recover a fixed accuracy per unit of added noise**, and show whether it **diverges** as you try to fully close the gap (the quantitative content the abstract promises and the current draft never plots).
- CV'd surface with coefficient CIs; saturating-vs-log model comparison table; per-`M` coefficient table superseding current Table A1.
- **Writing fixes (do regardless):**
  - **Correct the Plesner miscitation.** Current p.4 reads "false positives act as a form of reward perturbation that can regularize training (Plesner et al., 2026)." Plesner's regularization benefit is from *symmetric group-rollout* gradient *inversion* (flat-minima escape), **not** from false positives — and their directional finding is that **FP is the harmful direction**. As written you cite the one adjacent paper that contradicts you as if it supports you. Fix the attribution and instead cite Plesner as the contrasting prescription you reconcile.
  - State seeds/CIs on every number; replace "training reward plateaus" justification with the *true-accuracy* curves from §3.1.

```python
# --- iso-accuracy exchange rate (Phase 5) ---
# given fitted surface acc_hat(r, p, x), for a target accuracy a*:
#   along a noise ray (e.g. symmetric t=p=x), solve r(t) s.t. acc_hat(r, t, t) = a*
#   exchange rate = d log r / d t   ;  divergence ⇒ no finite-compute substitution
from scipy.optimize import brentq
def required_r(acc_hat, a_star, p, x, r_lo=4, r_hi=1e6):
    f = lambda r: acc_hat(r, p, x) - a_star
    return brentq(f, r_lo, r_hi) if f(r_lo)*f(r_hi) < 0 else np.inf  # inf ⇒ gap uncloseable
```

---

## 5. Risks & branch points (track these)
1. **Asymmetry not real** → Phase 0 Gate branch A′ (margin-interchangeability paper). Plan survives.
2. **Margin collapse fails** (accuracy does *not* collapse onto any P1–P3) → that is *itself* a publishable result ("signal strength alone doesn't predict RLVR-under-noise; structure dominates") and elevates the surface + moderators.
3. **Saturation undetectable even at r=128/256** → report bounded `A` estimates honestly; the asymptote test becomes a *lower bound* on the persistent gap (still answers the question, weaker form).
4. **Exchange rate finite and small** → compute *does* substitute cheaply; this contradicts the workshop framing but is a clean, citable number — report it straight, don't bury it.
5. **3B/7B compute infeasible** → keep model size as 2 points but downgrade Headline to "a law at small scale" and flag size-scaling as the explicit open question (don't overclaim "the" law).
6. **Qwen-math artifact (the §1.6 trap, restated as a tracked risk):** if math surfaces look flat/noise-robust on Qwen, do **not** report them alone — the cause may be Qwen's math prior (spurious-rewards), not noise tolerance. *Every* math run with a Qwen policy needs a non-Qwen counterpart or a same-model non-math counterpart before its numbers leave the lab. The *good* outcome here (4c) is a Qwen-vs-non-Qwen *divergence*, which is a debunking finding; the *bad* outcome is silently reporting a Qwen-prior effect as a noise effect. Never the latter.

---

## 6. Paper shape — pick the spine after Phase 0

- **If asymmetry real:** *"A dense FP×FN×compute×size sweep of RLVR under noisy verifiers, and the surface reveals that which error direction is more harmful is regime-dependent — reconciling apparently contradictory prior results as one mechanism."* Sweep = method; reconciliation + exchange rate = findings.
- **If asymmetry artifact:** *"RLVR accuracy under noisy verification is governed by a single signal-margin axis `m=1−p−x`; FP and FN are interchangeable at matched margin, and the compute↔verifier exchange rate is [number]."* Margin collapse = the result.
- **Bonus spine (if §4c fires):** *"Reported RLVR noise-robustness on math is partly an elicitation artifact of the Qwen pretraining prior, not a property of RL under noise — it does not transfer across model families."* This can run as a self-contained secondary contribution alongside either main spine, and is the high-ceiling payoff of putting math in deliberately.

Either way: the rigor (seeds, saturating fits, CV, wide range, size axis) is the **spine**; one of the two findings is the **"and therefore."** Don't ship the sweep as a measurement — give the reviewer the one sentence that makes it a finding.
