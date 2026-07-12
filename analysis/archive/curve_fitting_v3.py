import csv
import numpy as np
from numpy.linalg import lstsq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load all data ──
rows = []
with open("batch32_metrics.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["final_validation_accuracy"].strip() == "":
            continue
        rows.append(row)

run_names = [r["run_name"] for r in rows]
fp_all = np.array([float(r["false_positive_rate"]) for r in rows])
fn_all = np.array([float(r["false_negative_rate"]) for r in rows])
rollouts_all = np.array([float(r["num_rollouts"]) for r in rows])
val_acc_all = np.array([float(r["final_validation_accuracy"]) for r in rows])
model_size_all = np.array(["0.5B" if "0.5B" in name else "1.5B" for name in run_names])


def ols_fit(X, y):
    n = X.shape[0]
    X_aug = np.column_stack([np.ones(n), X])
    k = X_aug.shape[1]
    coeffs, _, _, _ = lstsq(X_aug, y, rcond=None)
    y_pred = X_aug @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    rmse = np.sqrt(ss_res / n)
    return coeffs, y_pred, r2, adj_r2, rmse


plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

for size in ["1.5B", "0.5B"]:
    mask = model_size_all == size
    fp = fp_all[mask]
    fn = fn_all[mask]
    r = rollouts_all[mask]
    y = val_acc_all[mask]
    lr = np.log2(r)
    sr = np.sqrt(r)
    n = len(y)

    print(f"\n{'='*85}")
    print(f"  {size} — fp, fn SEPARATE, quadratic max (n={n})")
    print(f"{'='*85}")

    results = []

    def reg(name, X, pred_fn_factory):
        c, pred, r2, adj_r2, rmse = ols_fit(X, y)
        pfn = pred_fn_factory(c)
        results.append((name, X.shape[1]+1, c, pred, r2, adj_r2, rmse, pfn))
        return c, r2, adj_r2, rmse

    # ── Linear ──
    reg("fp + fn + log₂r",
        np.column_stack([fp, fn, lr]),
        lambda c: lambda f, g, l: c[0] + c[1]*f + c[2]*g + c[3]*l)

    reg("fp + fn + √r",
        np.column_stack([fp, fn, sr]),
        lambda c: lambda f, g, l: c[0] + c[1]*f + c[2]*g + c[3]*np.sqrt(2**l))

    # ── Quadratic, no cross ──
    reg("fp² + fn² + fp + fn + log₂r",
        np.column_stack([fp**2, fn**2, fp, fn, lr]),
        lambda c: lambda f, g, l: c[0] + c[1]*f**2 + c[2]*g**2 + c[3]*f + c[4]*g + c[5]*l)

    reg("fp² + fn² + fp + fn + √r",
        np.column_stack([fp**2, fn**2, fp, fn, sr]),
        lambda c: lambda f, g, l: c[0] + c[1]*f**2 + c[2]*g**2 + c[3]*f + c[4]*g + c[5]*np.sqrt(2**l))

    # ── Quadratic + cross ──
    reg("fp² + fn² + fp·fn + fp + fn + log₂r",
        np.column_stack([fp**2, fn**2, fp*fn, fp, fn, lr]),
        lambda c: lambda f, g, l: c[0] + c[1]*f**2 + c[2]*g**2 + c[3]*f*g + c[4]*f + c[5]*g + c[6]*l)

    reg("fp² + fn² + fp·fn + fp + fn + √r",
        np.column_stack([fp**2, fn**2, fp*fn, fp, fn, sr]),
        lambda c: lambda f, g, l: c[0] + c[1]*f**2 + c[2]*g**2 + c[3]*f*g + c[4]*f + c[5]*g + c[6]*np.sqrt(2**l))

    # ── Quadratic + rollout interactions ──
    reg("fp² + fn² + fp·fn + fp + fn + log₂r + fp·log₂r + fn·log₂r",
        np.column_stack([fp**2, fn**2, fp*fn, fp, fn, lr, fp*lr, fn*lr]),
        lambda c: lambda f, g, l: c[0] + c[1]*f**2 + c[2]*g**2 + c[3]*f*g + c[4]*f + c[5]*g + c[6]*l + c[7]*f*l + c[8]*g*l)

    reg("fp² + fn² + fp·fn + fp + fn + √r + fp·√r + fn·√r",
        np.column_stack([fp**2, fn**2, fp*fn, fp, fn, sr, fp*sr, fn*sr]),
        lambda c: lambda f, g, l: c[0] + c[1]*f**2 + c[2]*g**2 + c[3]*f*g + c[4]*f + c[5]*g + c[6]*np.sqrt(2**l) + c[7]*f*np.sqrt(2**l) + c[8]*g*np.sqrt(2**l))

    # ── Quadratic no cross + rollout interactions ──
    reg("fp² + fn² + fp + fn + log₂r + fp·log₂r + fn·log₂r",
        np.column_stack([fp**2, fn**2, fp, fn, lr, fp*lr, fn*lr]),
        lambda c: lambda f, g, l: c[0] + c[1]*f**2 + c[2]*g**2 + c[3]*f + c[4]*g + c[5]*l + c[6]*f*l + c[7]*g*l)

    reg("fp² + fn² + fp + fn + √r + fp·√r + fn·√r",
        np.column_stack([fp**2, fn**2, fp, fn, sr, fp*sr, fn*sr]),
        lambda c: lambda f, g, l: c[0] + c[1]*f**2 + c[2]*g**2 + c[3]*f + c[4]*g + c[5]*np.sqrt(2**l) + c[6]*f*np.sqrt(2**l) + c[7]*g*np.sqrt(2**l))

    # ── Linear + rollout interactions ──
    reg("fp + fn + log₂r + fp·log₂r + fn·log₂r",
        np.column_stack([fp, fn, lr, fp*lr, fn*lr]),
        lambda c: lambda f, g, l: c[0] + c[1]*f + c[2]*g + c[3]*l + c[4]*f*l + c[5]*g*l)

    # ── Summary ──
    print(f"\n{'─'*90}")
    print(f"  {'Model':<60} {'k':>2} {'R²':>6} {'AdjR²':>7} {'RMSE':>7}")
    print(f"  {'─'*60} {'─'*2} {'─'*6} {'─'*7} {'─'*7}")
    for name, k, c, pred, r2, adj, rmse, pfn in sorted(results, key=lambda x: -x[5]):
        print(f"  {name:<60} {k:>2} {r2:>6.4f} {adj:>7.4f} {rmse:>7.4f}")

    best_by_adj = sorted(results, key=lambda x: -x[5])[0]
    print(f"\n  Best by Adj-R²: {best_by_adj[0]}")
    print(f"  k={best_by_adj[1]}, Adj-R²={best_by_adj[5]:.4f}, RMSE={best_by_adj[6]:.4f}")
    print(f"  Coefficients: {np.array2string(best_by_adj[2], precision=4, separator=', ')}")

    # Always plot the quadratic + cross model for both sizes
    target_name = "fp² + fn² + fp·fn + fp + fn + log₂r"
    best = [r for r in results if r[0] == target_name][0]
    print(f"\n  Plotting: {best[0]}")
    print(f"  k={best[1]}, Adj-R²={best[5]:.4f}, RMSE={best[6]:.4f}")
    print(f"  Coefficients: {np.array2string(best[2], precision=4, separator=', ')}")

    best_pfn = best[7]

    # Build equation string
    cc = best[2]
    eq_str = (f"y = {cc[0]:.3f} {cc[1]:+.3f}·fp² {cc[2]:+.3f}·fn² "
              f"{cc[3]:+.3f}·fp·fn {cc[4]:+.3f}·fp {cc[5]:+.3f}·fn "
              f"{cc[6]:+.4f}·log₂r")

    # ════════════════════════════════════════════════════
    # FIGURE 1: Heatmap fp vs fn — best model
    # ════════════════════════════════════════════════════
    fp_grid = np.linspace(0, 0.5, 60)
    fn_grid = np.linspace(0, 0.5, 60)
    FP, FN = np.meshgrid(fp_grid, fn_grid)

    rollout_plot_vals = [8, 16, 32]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{size}:  {eq_str}\n(Adj-R² = {best[5]:.3f},  RMSE = {best[6]:.4f})",
                 fontsize=13, y=1.04)

    vmin = 0.0 if size == "0.5B" else 0.2
    vmax = 0.6 if size == "0.5B" else 0.8

    for ax_idx, rv in enumerate(rollout_plot_vals):
        ax = axes[ax_idx]
        Z = best_pfn(FP, FN, np.log2(rv))
        im = ax.contourf(FP, FN, Z, levels=20, cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.contour(FP, FN, Z, levels=10, colors="k", linewidths=0.3, alpha=0.4)
        pts = mask & (rollouts_all == rv)
        ax.scatter(fp_all[pts], fn_all[pts], c=val_acc_all[pts], cmap="RdYlGn",
                   vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.8, s=60, zorder=5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("False Negative Rate")
        ax.set_title(f"r = {rv}")
        plt.colorbar(im, ax=ax, shrink=0.85, label="Val Accuracy")

    plt.tight_layout()
    plt.savefig(f"fig_v3_{size}_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved fig_v3_{size}_heatmap.png")

    # ════════════════════════════════════════════════════
    # FIGURE 2: Slices — acc vs fp (fixed fn) and acc vs fn (fixed fp)
    # ════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x_plot = np.linspace(0, 0.5, 300)
    rv_for_slice = 16

    ax = axes[0]
    for i, fn_val in enumerate([0.0, 0.1, 0.2, 0.3, 0.5]):
        y_fit = best_pfn(x_plot, fn_val, np.log2(rv_for_slice))
        ax.plot(x_plot, y_fit, color=colors[i], lw=2.5, alpha=0.8, label=f"fn={fn_val:.1f}")
        pts = mask & (np.abs(fn_all - fn_val) < 0.001) & (rollouts_all == rv_for_slice)
        if pts.any():
            ax.scatter(fp_all[pts], val_acc_all[pts], color=colors[i],
                       marker="o", s=70, edgecolors="k", linewidths=0.5, zorder=5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(f"Acc vs FP at r={rv_for_slice}")
    ax.legend(title="FN rate")
    ax.set_xlim(-0.02, 0.52)

    ax = axes[1]
    for i, fp_val in enumerate([0.0, 0.1, 0.2, 0.3, 0.5]):
        y_fit = best_pfn(fp_val, x_plot, np.log2(rv_for_slice))
        ax.plot(x_plot, y_fit, color=colors[i], lw=2.5, alpha=0.8, label=f"fp={fp_val:.1f}")
        pts = mask & (np.abs(fp_all - fp_val) < 0.001) & (rollouts_all == rv_for_slice)
        if pts.any():
            ax.scatter(fn_all[pts], val_acc_all[pts], color=colors[i],
                       marker="o", s=70, edgecolors="k", linewidths=0.5, zorder=5)
    ax.set_xlabel("False Negative Rate")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(f"Acc vs FN at r={rv_for_slice}")
    ax.legend(title="FP rate")
    ax.set_xlim(-0.02, 0.52)

    fig.suptitle(f"{size}:  {eq_str}\n(Adj-R² = {best[5]:.3f},  RMSE = {best[6]:.4f})",
                 fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig(f"fig_v3_{size}_slices.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved fig_v3_{size}_slices.png")

    # ════════════════════════════════════════════════════
    # FIGURE 3: Pred vs Actual
    # ════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    clr = "steelblue" if size == "1.5B" else "coral"
    ax.scatter(y, best[3], c=clr, alpha=0.5, edgecolors="k", linewidths=0.3, s=45)
    lims = [min(y.min(), best[3].min()) - 0.03, max(y.max(), best[3].max()) + 0.03]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Actual Validation Accuracy")
    ax.set_ylabel("Predicted Validation Accuracy")
    ax.set_title(f"{size}:  {eq_str}\nAdj-R² = {best[5]:.3f},  RMSE = {best[6]:.4f}", fontsize=11)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"fig_v3_{size}_pred_vs_actual.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved fig_v3_{size}_pred_vs_actual.png")
