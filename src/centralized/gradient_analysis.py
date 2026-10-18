"""
Per-step gradient statistics for the paper (put in src/centralized/).

  python -u -m centralized.gradient_analysis run --dataset har --imputation zero \
      --missing_rate 0.5 --fusion mid_late --seeds 50 121 893 17 2024
  python -u -m centralized.gradient_analysis summarize --dataset har --imputation zero \
      --missing_rate 0.5 --fusion mid_late

Logged at EVERY step that contains both clean and imputed samples:
  1-head : cos(grad_clean, grad_imputed) + both norms, at the classifier AND at the
           shared extractors+backbone (so it is comparable with the 2-head).
  2-head : same quantities at the shared extractors+backbone, aux loss NOT scaled by lambda
           (cosine is scale invariant; multiply the imputed norm by lambda for the effective one).
Gradients are of mean-reduced CE over each partition, so norms do not depend on partition size.
"""
import argparse, glob, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from centralized.config import DEVICE
from centralized.data_setup import get_dataloaders
from centralized.utils import get_dataset_details
from core.models import MultiTaskModel, SplitNetwork

OUT = "grad_logs"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def flat_grad(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    g = [x.reshape(-1) for x in g if x is not None]
    return torch.cat(g) if g else None


def stats(lc, li, params):
    gc, gi = flat_grad(lc, params), flat_grad(li, params)
    if gc is None or gi is None:
        return None
    return (F.cosine_similarity(gc, gi, dim=0).item(), gc.norm().item(), gi.norm().item())


def run_arch(arch, dataset, imp, rate, fusion, seed, a):
    nc, hd, bs, _, _, lm, lr = get_dataset_details(dataset)
    hd, bs, lm, lr = a.hidden_dim or hd, a.batch_size or bs, a.lmbda if a.lmbda is not None else lm, a.lr or lr
    set_seed(seed)
    train, _, _, dims = get_dataloaders(dataset, f"impute_{imp}", rate, seed, bs, use_validation=True)
    kw = {}
    if a.extractor_nonlinear: kw["extractor_nonlinear"] = True
    if a.backbone_depth is not None: kw["backbone_depth"] = a.backbone_depth
    set_seed(seed)
    cls = MultiTaskModel if arch == "2head" else SplitNetwork
    model = cls(dims, hd, nc, fusion, **kw).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    shared = list(model.extractors.parameters()) + list(model.backbone.parameters())
    cls_params = list(model.classifier.parameters())
    rows = []
    for ep in range(1, a.epochs + 1):
        model.train()
        for step, (x, y, aug) in enumerate(train):
            y = y.to(DEVICE); aug = aug.bool(); cl = ~aug
            both = bool(aug.any() and cl.any())
            opt.zero_grad()
            if arch == "1head":
                logits = model(({k: v.to(DEVICE) for k, v in x.items()}))
                loss = F.cross_entropy(logits, y)
                if both:
                    ac, ai = cl.to(DEVICE), aug.to(DEVICE)
                    lc, li = F.cross_entropy(logits[ac], y[ac]), F.cross_entropy(logits[ai], y[ai])
                    for level, p in (("classifier", cls_params), ("backbone", shared)):
                        s = stats(lc, li, p)
                        if s: rows.append((arch, level, ep, step, *s, int(cl.sum()), int(aug.sum())))
            else:
                loss = None; lp = la = None
                if cl.any():
                    lp = F.cross_entropy(model({k: v[cl].to(DEVICE) for k, v in x.items()}, use_auxiliary=False), y[cl.to(DEVICE)])
                    loss = lp
                if aug.any():
                    la = F.cross_entropy(model({k: v[aug].to(DEVICE) for k, v in x.items()}, use_auxiliary=True), y[aug.to(DEVICE)])
                    loss = lm * la if loss is None else loss + lm * la
                if both:
                    s = stats(lp, la, shared)
                    if s: rows.append((arch, "backbone", ep, step, *s, int(cl.sum()), int(aug.sum())))
            loss.backward(); opt.step()
    return rows


def cmd_run(a):
    os.makedirs(OUT, exist_ok=True)
    for seed in a.seeds:
        path = f"{OUT}/{a.dataset}_{a.imputation}_p{a.missing_rate}_{a.fusion}_seed{seed}.csv"
        if os.path.exists(path):
            print("skip", path); continue
        rows = []
        for arch in ("1head", "2head"):
            print(f"=== {a.dataset} {a.imputation} p={a.missing_rate} {a.fusion} {arch} seed={seed}", flush=True)
            rows += run_arch(arch, a.dataset, a.imputation, a.missing_rate, a.fusion, seed, a)
        pd.DataFrame(rows, columns=["arch", "level", "epoch", "step", "cos", "norm_clean",
                                    "norm_imp", "n_clean", "n_imp"]).to_csv(path, index=False)


def load(a):
    fs = sorted(glob.glob(f"{OUT}/{a.dataset}_{a.imputation}_p{a.missing_rate}_{a.fusion}_seed*.csv"))
    if not fs: raise SystemExit("no logs found")
    return pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(fs)])


def cmd_summarize(a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = load(a)
    d["neg"] = d.cos < 0
    d["ratio"] = d.norm_imp / d.norm_clean
    g = d.groupby(["arch", "level", "seed"]).agg(pct_neg=("neg", lambda s: 100 * s.mean()), mean_cos=("cos", "mean"),
                                                 med_ratio=("ratio", "median")).reset_index()
    t = g.groupby(["arch", "level"])[["pct_neg", "mean_cos", "med_ratio"]].agg(["mean", "std"])
    print(f"# {a.dataset} {a.imputation} p={a.missing_rate} {a.fusion} ({d.seed.nunique()} seeds, mean/std over seeds)")
    print(t.round(3).to_string())
    t.round(4).to_csv(f"{OUT}/summary_{a.dataset}_{a.imputation}_p{a.missing_rate}_{a.fusion}.csv")

    cols = {("1head", "classifier"): ("#2a78d6", "1-head, classifier"),
            ("1head", "backbone"): ("#eb6834", "1-head, backbone"),
            ("2head", "backbone"): ("#1baf7a", "2-head, backbone")}
    ep = d.groupby(["arch", "level", "seed", "epoch"])[["cos", "norm_clean", "norm_imp"]].mean().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for (arch, lvl), (c, lab) in cols.items():
        e = ep[(ep.arch == arch) & (ep.level == lvl)].groupby("epoch")
        for k, (col, key, ls) in enumerate([(0, "cos", "-")]):
            m, s = e[key].mean(), e[key].std().fillna(0)
            ax[0].plot(m.index, m, color=c, lw=2, label=lab)
            ax[0].fill_between(m.index, m - s, m + s, color=c, alpha=.15, lw=0)
        for key, ls in (("norm_clean", "-"), ("norm_imp", "--")):
            m = e[key].mean()
            ax[1].plot(m.index, m, color=c, lw=2, ls=ls)
    ax[0].axhline(0, color="#52514e", lw=.8)
    ax[0].set(xlabel="epoch", ylabel=r"cos($\nabla_{clean}$, $\nabla_{imputed}$)", title="Gradient alignment")
    ax[1].set(xlabel="epoch", ylabel="gradient norm", yscale="log", title="Gradient magnitude (solid: clean, dashed: imputed)")
    ax[0].legend(frameon=False, fontsize=8)
    for x in ax: x.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = f"{OUT}/fig_grad_{a.dataset}_{a.imputation}_p{a.missing_rate}_{a.fusion}.pdf"
    plt.savefig(out); print("saved", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    for n, f in (("run", cmd_run), ("summarize", cmd_summarize)):
        s = sp.add_parser(n); s.set_defaults(fn=f)
        s.add_argument("--dataset", required=True, choices=["modelnet", "sensit", "har", "youtube"])
        s.add_argument("--imputation", default="zero", choices=["zero", "gaus", "cpm", "dcp"])
        s.add_argument("--missing_rate", type=float, default=0.5)
        s.add_argument("--fusion", default="mid_late", choices=["early", "mid", "mid_late", "late"])
        if n == "run":
            s.add_argument("--seeds", type=int, nargs="+", default=[50, 121, 893, 17, 2024])
            s.add_argument("--epochs", type=int, default=20)
            s.add_argument("--lr", type=float); s.add_argument("--hidden_dim", type=int)
            s.add_argument("--batch_size", type=int); s.add_argument("--lmbda", type=float)
            s.add_argument("--extractor_nonlinear", action="store_true")
            s.add_argument("--backbone_depth", type=int)
    a = p.parse_args(); a.fn(a)
