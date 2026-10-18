"""
Hyperparameter search for the thesis experiments.

Two stages, both selecting on VALIDATION accuracy (never on test accuracy):

  stage 1 (--stage arch)   : grid over lr x hidden_dim (x batch_size) with the
                             1-head baseline. These are shared by 1-head and
                             2-head, so tuning them on the baseline keeps the
                             comparison fair (the 2-head model is not favoured).
  stage 2 (--stage lambda) : with the stage-1 winner fixed, grid over the
                             auxiliary loss weight lambda for the 2-head model.

Results are appended to a JSONL file (one line per run), so an interrupted
search resumes where it stopped. `summarize` builds the paper tables.

Usage (from src/):
  uv run python -u -m centralized.hparam_search search --stage arch   --dataset har
  uv run python -u -m centralized.hparam_search search --stage lambda --dataset har
  uv run python -u -m centralized.hparam_search summarize
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

from centralized.config import DEVICE
from centralized.data_setup import get_dataloaders
from centralized.train import train_and_eval, train_and_eval_mt
from centralized.utils import get_dataset_details
from core.models import MultiTaskModel, SplitNetwork


# --------------------------------------------------------------------------
# Search spaces per dataset (dynamically expanded to probe boundaries)
# --------------------------------------------------------------------------
def get_search_space(dataset: str):
    """Returns (lrs, hiddens, batches, lambdas) specific to the dataset."""
    lrs = [3e-4, 1e-3, 3e-3]
    hiddens = [32, 64, 128, 256]
    lambdas = [0.5, 1.0, 2.0, 3.0, 4.0]
    
    if dataset == "har":
        hiddens = [32, 64, 128, 256, 512]
        lambdas = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
    elif dataset == "sensit":
        lrs = [3e-4, 1e-3, 3e-3, 1e-2]
        # sensit lambda peaked perfectly at 3.0, no expansion needed
    elif dataset == "modelnet":
        hiddens = [32, 64, 128, 256, 512]
        lambdas = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
    elif dataset == "youtube":
        lrs = [1e-4, 3e-4, 1e-3, 3e-3]
        hiddens = [32, 64, 128, 256, 512]
        lambdas = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
        
    return lrs, hiddens, [32], lambdas


# Tuning seeds are deliberately different from the reporting seeds
# (50 121 893 17 2024) so the reported numbers are not tuned on themselves.
TUNE_SEEDS = [7, 11, 13]

DATASETS = ["har", "sensit", "modelnet", "youtube"]
RESULTS_FILE = "hparam_results/hparam_results.jsonl"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def val_accuracy(model, valloader, multitask: bool) -> float:
    """
    Validation accuracy of the model currently in memory. train_and_eval*
    reload the best-validation state before returning, so this equals the
    best validation accuracy of the run.
    """
    model.eval()
    correct, total = 0, 0
    for inputs_dict, labels, is_augmented in valloader:
        labels = labels.to(DEVICE)
        if multitask:
            aug = is_augmented.bool()
            clean = ~aug
            logits = torch.zeros(labels.size(0), model.num_classes, device=DEVICE)
            if clean.any():
                d = {k: v[clean].to(DEVICE) for k, v in inputs_dict.items()}
                logits[clean] = model(d, use_auxiliary=False)
            if aug.any():
                d = {k: v[aug].to(DEVICE) for k, v in inputs_dict.items()}
                logits[aug] = model(d, use_auxiliary=True)
        else:
            d = {k: v.to(DEVICE) for k, v in inputs_dict.items()}
            logits = model(d)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)


def load_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(run_key(json.loads(line)))
    return done


def run_key(r: dict):
    return (r["stage"], r["dataset"], r["model"], r["lr"], r["hidden_dim"],
            r["batch_size"], r["lmbda"], r["seed"], r["missing_rate"],
            r["fusion"], r["imputation"], r["epochs"])


def read_results(path: str) -> list[dict]:
    if not os.path.exists(path):
        sys.exit(f"No results file at {path}. Run `search` first.")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate(rows: list[dict], keys: tuple) -> dict:
    """Group rows by config and compute mean/std of validation accuracy."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r["val_acc"])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in groups.items()}


# --------------------------------------------------------------------------
# One training run
# --------------------------------------------------------------------------
def run_one(dataset, model_kind, lr, hidden_dim, batch_size, lmbda, seed,
            missing_rate, fusion, imputation, epochs, loader_cache):
    set_seed(seed)
    num_classes = get_dataset_details(dataset)[0]

    exp_type = f"impute_{imputation}"
    if model_kind == "multi":
        exp_type = f"2head_{exp_type}"

    cache_key = (dataset, batch_size, seed, missing_rate, imputation)
    if cache_key not in loader_cache:
        loader_cache.clear()  # keep memory bounded
        # use_validation=True: 15% of train is held out, test is untouched
        loader_cache[cache_key] = get_dataloaders(
            dataset, f"impute_{imputation}", missing_rate, seed,
            batch_size, use_validation=True
        )
    train_l, val_l, test_l, sensor_dims = loader_cache[cache_key]

    set_seed(seed)  # identical init for every config of the same seed
    if model_kind == "multi":
        model = MultiTaskModel(sensor_dims, hidden_dim, num_classes, fusion)
        train_and_eval_mt(model, train_l, test_l, valloader=val_l,
                          epochs=epochs, lr=lr, lmbda=lmbda)
    else:
        model = SplitNetwork(sensor_dims, hidden_dim, num_classes, fusion)
        train_and_eval(model, train_l, test_l, valloader=val_l,
                       epochs=epochs, lr=lr)

    return val_accuracy(model, val_l, multitask=(model_kind == "multi"))


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------
def best_arch(rows, dataset, fusion, imputation, missing_rate):
    """Stage-1 winner (lr, hidden_dim, batch_size) for a dataset."""
    sel = [r for r in rows if r["stage"] == "arch" and r["dataset"] == dataset
           and r["fusion"] == fusion and r["imputation"] == imputation
           and r["missing_rate"] == missing_rate]
    if not sel:
        sys.exit(f"No stage-1 results for {dataset}. Run `--stage arch` first.")
    agg = aggregate(sel, ("lr", "hidden_dim", "batch_size"))
    return max(agg.items(), key=lambda kv: kv[1][0])[0]


def cmd_search(args):
    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    done = load_done(args.results)
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    loader_cache: dict = {}

    for dataset in datasets:
        lrs, hiddens, batches, lambdas = get_search_space(dataset)
        
        if args.stage == "arch":
            grid = [(lr, hd, bs, 0.0) for lr, hd, bs in
                    itertools.product(lrs, hiddens, batches)]
            model_kind = "single"
        else:
            rows = read_results(args.results)
            lr, hd, bs = best_arch(rows, dataset, args.fusion,
                                   args.imputation, args.missing_rate)
            print(f"[{dataset}] stage-1 winner: lr={lr} hidden={hd} batch={bs}")
            grid = [(lr, hd, bs, lm) for lm in lambdas]
            model_kind = "multi"

        total = len(grid) * len(TUNE_SEEDS)
        i = 0
        for (lr, hd, bs, lm), seed in itertools.product(grid, TUNE_SEEDS):
            i += 1
            rec = dict(stage=args.stage, dataset=dataset, model=model_kind,
                       lr=lr, hidden_dim=hd, batch_size=bs, lmbda=lm, seed=seed,
                       missing_rate=args.missing_rate, fusion=args.fusion,
                       imputation=args.imputation, epochs=args.epochs)
            
            # This check guarantees we never re-run completed combinations
            if run_key(rec) in done:
                continue
                
            print(f"=== {dataset} | {args.stage} {i}/{total} | lr={lr} hd={hd} "
                  f"bs={bs} lambda={lm} seed={seed} : {time.strftime('%F %T')} ===",
                  flush=True)
            rec["val_acc"] = run_one(
                dataset, model_kind, lr, hd, bs, lm, seed, args.missing_rate,
                args.fusion, args.imputation, args.epochs, loader_cache)
            with open(args.results, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done.add(run_key(rec))
            print(f"-> val acc: {rec['val_acc']*100:.2f}%", flush=True)


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------
def fmt_list(vals):
    return "{" + ", ".join(f"{v:g}" for v in vals) + "}"


def cmd_summarize(args):
    rows = read_results(args.results)
    out = []
    table = []

    for dataset in DATASETS:
        lrs, hiddens, batches, lambdas = get_search_space(dataset)
        
        sel1 = [r for r in rows if r["stage"] == "arch" and r["dataset"] == dataset]
        if not sel1:
            continue
        agg1 = aggregate(sel1, ("lr", "hidden_dim", "batch_size"))
        ranked = sorted(agg1.items(), key=lambda kv: -kv[1][0])
        (lr, hd, bs), (m, s, n) = ranked[0]
        gap = (m - ranked[1][1][0]) * 100 if len(ranked) > 1 else float("nan")

        lam_txt, lam_m = "-", None
        sel2 = [r for r in rows if r["stage"] == "lambda" and r["dataset"] == dataset]
        if sel2:
            agg2 = aggregate(sel2, ("lmbda",))
            (lam,), (lam_m, lam_s, _) = max(agg2.items(), key=lambda kv: kv[1][0])
            lam_txt = f"{lam:g}"

        default = get_dataset_details(dataset)
        table.append((dataset, lr, hd, bs, lam_txt, m, s, n, gap, default, lrs, hiddens, lambdas))

        out.append(f"\n### {dataset}: stage 1 (val acc, mean +/- std over {n} seeds)")
        out.append("| lr | hidden_dim | batch | val acc |")
        out.append("|---|---|---|---|")
        for (l, h, b), (mm, ss, _) in ranked:
            mark = " **<-**" if (l, h, b) == (lr, hd, bs) else ""
            out.append(f"| {l:g} | {h} | {b} | {mm*100:.2f} +/- {ss*100:.2f}{mark} |")
        if sel2:
            out.append(f"\n{dataset}: stage 2 (lambda, 2-head)")
            out.append("| lambda | val acc |")
            out.append("|---|---|")
            for (lm,), (mm, ss, _) in sorted(agg2.items()):
                out.append(f"| {lm:g} | {mm*100:.2f} +/- {ss*100:.2f} |")

    # Paper table (markdown)
    head = ("| Dataset | Search space (lr, hidden_dim, lambda) | Selected lr | "
            "Selected hidden_dim | Selected lambda | Val acc (1-head) | Gap to runner-up |")
    sep = "|---|---|---|---|---|---|---|"
    md = [head, sep]
    for d, lr, hd, bs, lam, m, s, n, gap, _, lrs, hiddens, lambdas in table:
        space = f"lr {fmt_list(lrs)}; hidden {fmt_list(hiddens)}; lambda {fmt_list(lambdas)}"
        md.append(f"| {d} | {space} | {lr:g} | {hd} | {lam} | "
                  f"{m*100:.2f} +/- {s*100:.2f} | {gap:.2f} pp |")

    # LaTeX
    tex = [r"\begin{tabular}{llcccc}", r"\toprule",
           r"Dataset & Search space & lr & hidden dim & $\lambda$ & Val. acc. (\%) \\",
           r"\midrule"]
    for d, lr, hd, bs, lam, m, s, n, gap, _, lrs, hiddens, lambdas in table:
        tex.append(f"{d} & lr $\\in$ {fmt_list(lrs)}, $h \\in$ {fmt_list(hiddens)}, "
                   f"$\\lambda \\in$ {fmt_list(lambdas)} & {lr:g} & {hd} & {lam} & "
                   f"${m*100:.2f} \\pm {s*100:.2f}$ \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]

    text = "\n".join(["## Paper table (markdown)", *md, "", "## Paper table (LaTeX)",
                      *tex, "", "## Full grids", *out])

    print(text)
    path = os.path.join(os.path.dirname(args.results) or ".", "hparam_summary.md")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved to {path}")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search")
    s.add_argument("--stage", choices=["arch", "lambda"], required=True)
    s.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    s.add_argument("--fusion", choices=["early", "mid", "mid_late", "late"],
                   default="mid_late")
    s.add_argument("--imputation", choices=["zero", "gaus"], default="zero",
                   help="zero/gaus only: cpm/dcp ignore the overridden hidden_dim")
    s.add_argument("--missing_rate", type=float, default=0.5)
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--results", default=RESULTS_FILE)
    s.set_defaults(fn=cmd_search)

    t = sub.add_parser("summarize")
    t.add_argument("--results", default=RESULTS_FILE)
    t.set_defaults(fn=cmd_summarize)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
