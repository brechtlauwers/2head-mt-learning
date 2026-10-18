"""
Per-seed accuracies, paired significance tests, gain-vs-MSE figure (put in src/centralized/).

  # 1) re-run everything and keep per-seed accuracies (resumable; uses the hyperparameters in utils.py)
  python -u -m centralized.paper_stats collect --datasets har sensit modelnet youtube \
      --imputations zero gaus cpm dcp --fusions mid_late --rates 0.1 0.3 0.5 0.7 --seeds 50 121 893 17 2024
  # 2) paired t-test per cell, Holm-corrected over all cells
  python -m centralized.paper_stats table
  # 3) gain vs. imputation MSE (needs mse_results.json from imputation_mse.py)
  python -m centralized.paper_stats gain_vs_mse --fusion mid_late
"""
import argparse, json, os
import numpy as np

FILE = "per_seed_results.jsonl"


def cmd_collect(a):
    from centralized.experiments import run_experiment
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l); done.add((r["dataset"], r["imputation"], r["fusion"], r["rate"], r["seed"], r["arch"]))
    for ds in a.datasets:
        for imp in a.imputations:
            for f in a.fusions:
                for rate in a.rates:
                    for seed in a.seeds:
                        for arch, pre in (("1head", "impute_"), ("2head", "2head_impute_")):
                            key = (ds, imp, f, rate, seed, arch)
                            if key in done: continue
                            acc = run_experiment(ds, pre + imp, rate, seed, fusion_type=f)
                            with open(a.out, "a") as fh:
                                fh.write(json.dumps(dict(dataset=ds, imputation=imp, fusion=f, rate=rate,
                                                         seed=seed, arch=arch, acc=acc)) + "\n")


def cells(path):
    d = {}
    for l in open(path):
        r = json.loads(l)
        d.setdefault((r["dataset"], r["imputation"], r["fusion"], r["rate"]), {}).setdefault(r["arch"], {})[r["seed"]] = r["acc"] * 100
    out = {}
    for k, v in d.items():
        if "1head" in v and "2head" in v:
            s = sorted(set(v["1head"]) & set(v["2head"]))
            out[k] = (np.array([v["1head"][i] for i in s]), np.array([v["2head"][i] for i in s]))
    return out


def holm(p):
    o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(run, 1)
    return adj


def cmd_table(a):
    from scipy.stats import ttest_rel
    c = cells(a.inp); keys = sorted(c)
    p = np.array([ttest_rel(c[k][1], c[k][0]).pvalue if len(c[k][0]) > 1 and np.ptp(c[k][1] - c[k][0]) > 0 else 1.0 for k in keys])
    ph = holm(p)
    print("| Dataset | Imputation | Fusion | Rate | n | 1H (mean±std) | 2H (mean±std) | Δ | p | p_Holm |")
    for k, pv, pa in zip(keys, p, ph):
        x1, x2 = c[k]
        # np.std (ddof=0) to match the existing tables
        print(f"| {k[0]} | {k[1]} | {k[2]} | {int(k[3]*100)}% | {len(x1)} | {x1.mean():.2f} ± {x1.std():.2f} | "
              f"{x2.mean():.2f} ± {x2.std():.2f} | {x2.mean()-x1.mean():+.2f} | {pv:.3f} | {pa:.3f} |")
    print("\nNote: with n seeds a paired t-test has little power; Wilcoxon cannot go below p=2/2^n (n=5: 0.0625).")


def cmd_gain(a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    c, mse = cells(a.inp), json.load(open(a.mse))
    col = {"modelnet": "#2a78d6", "sensit": "#eb6834", "har": "#1baf7a", "youtube": "#6250d6"}
    mk = {0.1: "o", 0.3: "s", 0.5: "^", 0.7: "D"}
    xs, ys = [], []
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for (ds, imp, f, r), (x1, x2) in c.items():
        if f != a.fusion or r == 0 or imp not in mse.get(ds, {}): continue
        x, y = mse[ds][imp]["mean"], x2.mean() - x1.mean()
        xs.append(x); ys.append(y)
        ax.scatter(x, y, color=col[ds], marker=mk[r], s=36, edgecolor="white", lw=.8)
    lx = np.log10(xs); b, a0 = np.polyfit(lx, ys, 1); g = np.linspace(lx.min(), lx.max(), 50)
    ax.plot(10 ** g, b * g + a0, color="#52514e", lw=1.5)
    rho, pv = spearmanr(xs, ys)
    ax.set(xscale="log", xlabel="imputation MSE (50% missing, test)", ylabel="two-head gain Δ (pp)",
           title=f"{a.fusion}: Spearman ρ={rho:.2f} (p={pv:.3f}, n={len(xs)})")
    ax.axhline(0, color="#52514e", lw=.6, ls=":")
    for ds in col: ax.scatter([], [], color=col[ds], label=ds)
    for r in mk: ax.scatter([], [], color="#52514e", marker=mk[r], label=f"{int(r*100)}%")
    ax.legend(frameon=False, fontsize=7, ncol=2); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig("fig_gain_vs_mse.pdf"); print(f"rho={rho:.3f} p={pv:.4f} slope/decade={b:.3f}")
    print("NOTE: MSE is measured at one rate; if you plot other rates the x-values are re-used as-is.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("collect"); s.set_defaults(fn=cmd_collect)
    s.add_argument("--datasets", nargs="+", default=["modelnet", "sensit", "har", "youtube"])
    s.add_argument("--imputations", nargs="+", default=["zero", "gaus", "cpm", "dcp"])
    s.add_argument("--fusions", nargs="+", default=["mid_late"])
    s.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5, 0.7])
    s.add_argument("--seeds", type=int, nargs="+", default=[50, 121, 893, 17, 2024])
    s.add_argument("--out", default=FILE)
    s = sp.add_parser("table"); s.set_defaults(fn=cmd_table); s.add_argument("--inp", default=FILE)
    s = sp.add_parser("gain_vs_mse"); s.set_defaults(fn=cmd_gain)
    s.add_argument("--inp", default=FILE); s.add_argument("--mse", default="mse_results.json")
    s.add_argument("--fusion", default="mid_late")
    a = p.parse_args(); a.fn(a)
