"""Generate figures for the RLVR noisy-verifier research notebook.
All values are the recorded n=5 converged numbers (see logs/*.md) — reproducible, no W&B pull.
Run: uv run --no-sync python notebook/make_figures.py  (writes notebook/figures/*.png)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "notebook/figures"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

# ---------- Fig 1: asymmetry vs base rate (the money figure) ----------
# (config, base b, FP(.3,0), FN(0,.3)); n=5 each
cfg = [("OLMo-MATH",0.121,0.029,0.080,"o"),("MATH-1.5B*",0.422,0.216,0.381,"o"),
       ("GSM8K-0.5B",0.525,0.490,0.481,"s"),("OLMo-GSM8K",0.577,0.515,0.512,"s"),
       ("MATH-0.5B",0.591,0.570,0.580,"s"),("MATH-3B",0.728,0.717,0.723,"s"),
       ("GSM8K-1.5B",0.762,0.730,0.738,"s"),("GSM8K-3B",0.841,0.809,0.805,"s"),
       ("MATH-1.5B lv1,2",0.876,0.833,0.848,"s"),
       # code = EXECUTION verifier (2nd verifier type), triangle markers
       ("code-0.5B-MBPP",0.404,0.335,0.329,"^"),("code-1.5B-MBPP",0.508,0.491,0.507,"^")]
b = np.array([c[1] for c in cfg]); fp=np.array([c[2] for c in cfg]); fn=np.array([c[3] for c in cfg])
ratio = fp/fn; diff = fp-fn
fig,ax = plt.subplots(1,2,figsize=(11,4.2))
for i,c in enumerate(cfg):
    col = "crimson" if ratio[i]<0.9 else "steelblue"
    ax[0].scatter(b[i],ratio[i],c=col,s=64,zorder=3,marker=c[4])
    ax[0].annotate(c[0],(b[i],ratio[i]),fontsize=6.5,xytext=(4,4),textcoords="offset points")
ax[0].scatter([],[],c="gray",marker="^",label="code (execution verifier)")
ax[0].scatter([],[],c="gray",marker="o",label="math (equivalence verifier)"); ax[0].legend(fontsize=7,loc="lower right")
ax[0].axhline(1.0,ls="--",c="gray",lw=1); ax[0].axvspan(0.45,0.5,color="orange",alpha=0.15)
ax[0].set_xlabel("current precision (clean-eval accuracy)"); ax[0].set_ylabel("FP/FN accuracy ratio")
ax[0].set_title("Asymmetry (ratio) vs precision\nmonotone; sharp threshold ≈0.45–0.5")
ax[0].text(0.47,0.42,"threshold",rotation=90,fontsize=8,color="darkorange",va="bottom")
for i in range(len(cfg)):
    col="crimson" if diff[i]<-0.03 else "steelblue"
    ax[1].scatter(b[i],diff[i],c=col,s=60,zorder=3)
ax[1].axhline(0,ls="--",c="gray",lw=1); ax[1].axvspan(0.45,0.5,color="orange",alpha=0.15)
ax[1].set_xlabel("current precision (clean-eval accuracy)"); ax[1].set_ylabel("FP − FN accuracy")
ax[1].set_title("Absolute FP−FN vs precision\n(hump = [0,1] floor artifact; ratio is cleaner)")
fig.suptitle("Fig 1. FP/FN asymmetry is PRECISION-moderated (2 tasks × 3 sizes × 2 families)\n"
             "*MATH-1.5B 0.42 = undertrained snapshot (epoch trap); see Fig 6 for the within-run trajectory",fontweight="bold",fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_asymmetry_vs_baserate.png",bbox_inches="tight"); plt.close(fig)

# ---------- Fig 2: margin-collapse (GSM8K) vs break (MATH) ----------
# GSM8K-1.5B marginals: FP-only (fn=0) and FN-only (fp=0), acc vs m=1-rate
rate=[0.1,0.2,0.3,0.4,0.5]; m=[1-r for r in rate]
gsm_fp=[0.739,0.734,0.730,0.717,0.698]; gsm_fn=[0.740,0.726,0.738,0.698,0.705]
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
ax[0].plot(m,gsm_fp,"o-",label="FP only (fn=0)",c="crimson")
ax[0].plot(m,gsm_fn,"s--",label="FN only (fp=0)",c="steelblue")
ax[0].set_title("GSM8K-1.5B (base 0.76): COLLAPSE\nFP & FN overlap → acc = f(m)")
ax[0].set_xlabel("margin m = 1 − fp − fn"); ax[0].set_ylabel("accuracy"); ax[0].legend(); ax[0].invert_xaxis()
# MATH-1.5B at m=0.7: FP vs FN split
ax[1].bar([0,1],[0.216,0.381],color=["crimson","steelblue"],width=0.5)
ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["FP (0.3,0)","FN (0,0.3)"])
ax[1].set_ylabel("accuracy"); ax[1].set_ylim(0,0.5)
ax[1].set_title("MATH-1.5B (base 0.42): BREAK\nsame m=0.7, FP≪FN (−0.166)")
for x,y in [(0,0.216),(1,0.381)]: ax[1].text(x,y+0.01,f"{y:.3f}",ha="center")
fig.suptitle("Fig 2. Margin-collapse holds at high base rate, breaks (FP-worse) at low",fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_margin_collapse.png",bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: compute axis (saturation + noise offset) ----------
rc=[4,8,16,32,64,128,256]; yc=[0.684,0.709,0.734,0.762,0.778,0.781,0.804]
rn_=[4,16,32,64,128];  yn=[0.643,0.673,0.693,0.738,0.727]
fig,ax=plt.subplots(figsize=(6.4,4.4))
ax.plot(rc,yc,"o-",c="steelblue",label="clean")
ax.plot(rn_,yn,"s--",c="crimson",label="noisy (fp=fn=0.2)")
ax.text(5,0.70,"per-doubling gain:\n~0.025 (r$\\leq$32) $\\to$ ~0.015 (r$\\geq$64)\nstrong diminishing returns;\nstill creeping to ~0.80 at r=256\n(plateau vs slow-rise within $\\pm$0.02 noise)",fontsize=7,color="gray")
ax.set_xscale("log",base=2); ax.set_xticks(rc); ax.set_xticklabels(rc)
ax.set_xlabel("rollouts per prompt  r  (log2)"); ax.set_ylabel("accuracy")
ax.set_title("Fig 3. Compute axis (Qwen-1.5B GSM8K)\nlog-linear (r$\\leq$32) $\\to$ strong diminishing returns ($\\sim$0.80); noise = small const offset",fontweight="bold")
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_compute.png",bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4: phase transition across full margin (extended range) ----------
# GSM8K-1.5B, acc vs m (FP/FN averaged where symmetric); m from +1 to -0.8
mm=[1.0,0.7,0.3,0.1,0.0,-0.4,-0.8]
acc=[0.762,0.730,0.672,0.575,0.031,0.004,0.000]
fig,ax=plt.subplots(figsize=(6.4,4.4))
ax.plot(mm,acc,"o-",c="darkgreen")
ax.axvline(0,ls="--",c="red",lw=1); ax.axhline(0.027,ls=":",c="gray",lw=1)
ax.text(-0.75,0.05,"init ~0.03",fontsize=8,color="gray")
ax.text(0.05,0.4,"m=0: no signal\n(frozen)",fontsize=8,color="red")
ax.annotate("learn",(0.7,0.73),fontsize=9,color="darkgreen")
ax.annotate("learn-to-be-WRONG",(-0.6,0.06),fontsize=8,color="purple")
ax.set_xlabel("margin m = 1 − fp − fn  (fp,fn up to 1)"); ax.set_ylabel("accuracy")
ax.set_title("Fig 4. Phase transition in the margin (GSM8K-1.5B)\nlearn (m>0) → frozen (m=0) → anti-learn (m<0)",fontweight="bold")
ax.invert_xaxis()
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_phase_transition.png",bbox_inches="tight"); plt.close(fig)

# ---------- Fig 5: A×B — noise gap vs compute, 0.5B vs 1.5B ----------
r15=[4,16,32,64,128]; gap15=[0.041,0.062,0.069,0.041,0.055]
r05=[8,32,128];       gap05=[0.148,0.112,0.065]
fig,ax=plt.subplots(figsize=(6.2,4.4))
ax.plot(r15,gap15,"o-",c="steelblue",label="1.5B (high base) — gap ~constant")
ax.plot(r05,gap05,"s--",c="crimson",label="0.5B (low base) — gap shrinks")
ax.set_xscale("log",base=2); ax.set_xticks([4,8,16,32,64,128]); ax.set_xticklabels([4,8,16,32,64,128])
ax.set_xlabel("rollouts per prompt  r  (log2)"); ax.set_ylabel("clean − noisy accuracy gap")
ax.set_title("Fig 5. A×B link: does compute buy back noise?\nyes at low base (0.5B: 0.148$\\to$0.065), no at high base (1.5B: ~const)",fontweight="bold")
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_AxB_transfer.png",bbox_inches="tight"); plt.close(fig)

# ---------- Fig 6: the ESCAPE — asymmetry is a low-precision TRANSIENT (n=5 trajectories) ----------
# MATH-1.5B, difficulty bins mL4(lv1-4)/mL5(lv1-5), clean/FP(.3,0)/FN(0,.3), r=32, val every 5 steps.
# As precision climbs through training the model ESCAPES the low-precision trap and FP catches up → FP-FN→0.
mL4_clean=[0.0195,0.0446,0.082,0.232,0.3812,0.5149,0.6086,0.6656,0.6906,0.7305,0.75,0.7601,0.7703,0.775,0.7625,0.7766,0.7664,0.7726,0.7625,0.75,0.7719,0.7555,0.75,0.7625,0.7656,0.7531,0.7633,0.7594,0.7539,0.7625,0.7539,0.7773,0.7555,0.7523]
mL4_fp=[0.0195,0.0414,0.0539,0.0789,0.1703,0.2742,0.3523,0.5109,0.5945,0.6508,0.6648,0.6984,0.7117,0.7227,0.7383,0.757,0.7344,0.732,0.7266,0.736,0.7375,0.7195,0.7242,0.7484,0.7414,0.7383,0.7375,0.7586,0.7477,0.7305,0.7453,0.7289,0.7383,0.7453]
mL4_fn=[0.0195,0.0485,0.0656,0.1914,0.3469,0.4477,0.5867,0.6422,0.6867,0.6984,0.732,0.7219,0.7492,0.7469,0.7703,0.7562,0.7399,0.7336,0.7328,0.7398,0.7453,0.7352,0.7383,0.7344,0.7281,0.7359,0.7406,0.7383,0.7313,0.725,0.7594,0.7453,0.7524,0.7367]
mL5_clean=[0.0234,0.0281,0.0555,0.1711,0.3101,0.4086,0.4797,0.5469,0.5867,0.6258,0.6422,0.6734,0.6719,0.6586,0.6828,0.6633,0.6781,0.6648,0.6789,0.6562,0.6492,0.6789,0.6633,0.6649,0.6703,0.6852,0.6828,0.6672,0.668,0.6688,0.6735,0.6688,0.675,0.6781,0.6836]
mL5_fp=[0.0234,0.0422,0.0461,0.0547,0.1023,0.1922,0.2805,0.3992,0.4563,0.5258,0.5563,0.5914,0.6094,0.6203,0.6195,0.6492,0.6336,0.639,0.6187,0.6195,0.6273,0.6352,0.6633,0.657,0.6328,0.6437,0.6274,0.6609,0.6469,0.6406,0.6453,0.643,0.6453,0.6445]
mL5_fn=[0.0234,0.0328,0.0594,0.1484,0.2906,0.3766,0.4594,0.5242,0.5445,0.5812,0.6297,0.6226,0.643,0.6461,0.6469,0.6602,0.6695,0.6531,0.6398,0.6312,0.6414,0.6484,0.6445,0.6672,0.6539,0.6703,0.6508,0.6437,0.6406,0.6211,0.6211,0.6352,0.6383,0.6476,0.6399,0.6476,0.6367,0.6594]
def _s(a): return [5*i for i in range(len(a))]
fig,ax=plt.subplots(1,2,figsize=(11.5,4.4))
# left: accuracy trajectories (mL4) — FP catches up
ax[0].plot(_s(mL4_clean),mL4_clean,"-",c="gray",label="clean")
ax[0].plot(_s(mL4_fp),mL4_fp,"-",c="crimson",label="FP (0.3,0)")
ax[0].plot(_s(mL4_fn),mL4_fn,"--",c="steelblue",label="FN (0,0.3)")
ax[0].axvspan(0,45,color="crimson",alpha=0.07); ax[0].text(6,0.30,"trapped\n(low precision):\nFP $\\ll$ FN",fontsize=8,color="crimson")
ax[0].text(115,0.55,"escaped:\nFP $\\approx$ FN",fontsize=8,color="gray")
ax[0].set_xlabel("training step"); ax[0].set_ylabel("clean-eval accuracy")
ax[0].set_title("MATH-1.5B lv1-4 (n=5): FP catches up as\nprecision climbs — the escape",fontsize=10); ax[0].legend(fontsize=8,loc="lower right")
# right: FP-FN vs current precision — pool BOTH bins' trajectories + converged/snapshot configs
for cl,fpc,fnc,lab,col in [(mL4_clean,mL4_fp,mL4_fn,"mL4 traj","crimson"),(mL5_clean,mL5_fp,mL5_fn,"mL5 traj","darkorange")]:
    L=min(len(cl),len(fpc),len(fnc))
    ax[1].plot([cl[i] for i in range(L)],[fpc[i]-fnc[i] for i in range(L)],".-",c=col,ms=4,lw=1,alpha=0.8,label=lab)
# converged/snapshot config points (precision=clean base, FP-FN); marker o=math ^=code
pts=[("OLMo-MATH",0.121,-0.051,"o"),("lv1-3*",0.422,-0.165,"o"),("GSM-0.5B",0.525,0.009,"o"),
     ("MATH-0.5B",0.591,-0.010,"o"),("MATH-3B",0.728,-0.006,"o"),("GSM-1.5B",0.762,-0.008,"o"),
     ("GSM-3B",0.841,0.004,"o"),("lv1-2",0.876,-0.015,"o"),("code-0.5B",0.404,0.006,"^"),("code-1.5B",0.508,-0.016,"^")]
for nm,p,d,mk in pts:
    ax[1].scatter(p,d,c="black",s=45,marker=mk,zorder=5)
    ax[1].annotate(nm,(p,d),fontsize=6,xytext=(3,3),textcoords="offset points")
ax[1].axhline(0,ls="--",c="gray",lw=1); ax[1].axvspan(0,0.45,color="crimson",alpha=0.07)
ax[1].text(0.02,-0.15,"low-precision\ntrap (FP-worse)",fontsize=8,color="crimson")
ax[1].set_xlabel("current precision (clean-eval accuracy)"); ax[1].set_ylabel("FP − FN accuracy")
ax[1].set_title("Asymmetry vs PRECISION: FP-worse concentrated at low\nprecision/early training; magnitude task-dependent (MATH$\\gg$code)",fontsize=10)
ax[1].legend(fontsize=7,loc="lower right")
fig.suptitle("Fig 6. The FP-worse asymmetry is a low-precision TRANSIENT, not a converged property\n(*lv1-3 0.42 was an undertrained snapshot; longer training escapes → symmetric)",fontweight="bold",fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_escape.png",bbox_inches="tight"); plt.close(fig)

print("wrote 6 figures to", OUT)
