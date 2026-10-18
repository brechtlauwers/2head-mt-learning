"""
Imputation MSE on the test set (put in src/centralized/).
  python -u -m centralized.imputation_mse --datasets har sensit modelnet youtube --missing_rate 0.5 --seeds 50 121 893
Writes mse_results.json: {dataset: {imputation: {"mean": .., "std": .., "per_seed": [..]}}}
MSE is per imputed view, averaged over feature dims and over all imputed views of the test set,
on the L2-normalised features the models see. It reuses the cached CPM/DCP weights (use_validation=True).
"""
import argparse, json
import numpy as np
from centralized.data_setup import get_dataloaders


def mse_for(dataset, imp, rate, seed):
    _, _, test, _ = get_dataloaders(dataset, f"impute_{imp}", rate, seed, 32, use_validation=True)
    w = test.dataset                      # wrapper with .base_dataset and .missing_matrix
    miss = w.missing_matrix
    tot, n = 0.0, 0
    for i in range(len(w)):
        if not miss[:, i].any():
            continue
        imputed, _, _ = w[i]
        true, _ = w.base_dataset[i]
        for v in np.where(miss[:, i])[0]:
            k = f"view_{v}"
            tot += float(((imputed[k] - true[k]) ** 2).mean()); n += 1
    return tot / max(n, 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["modelnet", "sensit", "har", "youtube"])
    p.add_argument("--imputations", nargs="+", default=["zero", "gaus", "cpm", "dcp"])
    p.add_argument("--missing_rate", type=float, default=0.5)
    p.add_argument("--seeds", type=int, nargs="+", default=[50, 121, 893])
    p.add_argument("--out", default="mse_results.json")
    a = p.parse_args()
    res = {}
    for ds in a.datasets:
        res[ds] = {}
        for imp in a.imputations:
            v = [mse_for(ds, imp, a.missing_rate, s) for s in a.seeds]
            res[ds][imp] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v}
            print(f"{ds:9s} {imp:5s} MSE {np.mean(v):.5f} +/- {np.std(v):.5f}", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
