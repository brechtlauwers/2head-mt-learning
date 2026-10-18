#!/usr/bin/env bash
set -uo pipefail
source ~/.bashrc
export MPLBACKEND=Agg
LOGDIR="grad_logs"
mkdir -p $LOGDIR
unset DISPLAY


DATASETS="youtube"
IMPS="zero gaus cpm dcp"
RATES="0.1 0.3 0.5 0.7"
SEEDS="50 121 893 17 2024"
GRAD_SEEDS="50 121 893"
RESULTS=per_seed_results_v2.jsonl
PY="uv run python -u -m"

FAILED=()
stamp() { echo "=== $* : $(date) ==="; }
step() { local name=$1; shift; stamp "$name"; "$@" || FAILED+=("$name"); }

# ---------- 0. preflight ----------
stamp "preflight"
TMP=$(mktemp -d)
$PY centralized.paper_stats collect --datasets youtube --imputations zero --fusions mid --rates 0.5 \
    --seeds 50 --out "$TMP/pre.jsonl"                                  || { echo "PREFLIGHT FAILED: paper_stats"; exit 1; }
$PY centralized.imputation_mse --datasets youtube --imputations zero --seeds 50 --out "$TMP/mse.json" \
                                                                       || { echo "PREFLIGHT FAILED: mse"; exit 1; }
$PY centralized.gradient_analysis run --dataset youtube --imputation zero --missing_rate 0.123 --seeds 50 --epochs 1 --fusion mid \
                                                                       || { echo "PREFLIGHT FAILED: grad"; exit 1; }
rm -f grad_logs/youtube_zero_p0.123_mid_seed50.csv
echo "preflight OK"

# ---------- 1. main results (per-seed accuracies) ----------
step "main 0% reference" $PY centralized.paper_stats collect --datasets $DATASETS --imputations zero \
    --fusions early mid late --rates 0.0 --seeds $SEEDS --out $RESULTS

step "main all missing rates" $PY centralized.paper_stats collect --datasets $DATASETS --imputations $IMPS \
    --fusions early mid late --rates $RATES --seeds $SEEDS --out $RESULTS

step "paired tests" bash -c "$PY centralized.paper_stats table --inp $RESULTS > $LOGDIR/paired_table.md"

# ---------- 2. imputation MSE ----------
step "imputation mse" $PY centralized.imputation_mse --datasets $DATASETS --imputations $IMPS \
    --missing_rate 0.5 --seeds $GRAD_SEEDS --out mse_results_v2.json
step "gain vs mse" $PY centralized.paper_stats gain_vs_mse --inp $RESULTS --mse mse_results_v2.json --fusion mid

# ---------- 3. gradient analysis ----------
for ds in $DATASETS; do
  for imp in $IMPS; do
    step "grad $ds $imp 0.5" $PY centralized.gradient_analysis run --dataset $ds --imputation $imp \
        --missing_rate 0.5 --fusion mid --seeds $GRAD_SEEDS
    step "grad summary $ds $imp 0.5" bash -c "$PY centralized.gradient_analysis summarize --dataset $ds --imputation $imp --missing_rate 0.5 --fusion mid | tee $LOGDIR/gradsum_${ds}_${imp}_0.5.txt"
  done
  for r in 0.1 0.3 0.7; do
    step "grad $ds zero $r" $PY centralized.gradient_analysis run --dataset $ds --imputation zero \
        --missing_rate $r --fusion mid --seeds $GRAD_SEEDS
    step "grad summary $ds zero $r" bash -c "$PY centralized.gradient_analysis summarize --dataset $ds --imputation zero --missing_rate $r --fusion mid | tee $LOGDIR/gradsum_${ds}_zero_${r}.txt"
  done
done

# ---------- 4. depth ablation ----------
depth_run() {
  local ds=$1 depth=$2 fusion=$3 log="$LOGDIR/arch_${1}_depth${2}_${3}.log"
  if [ -s "$log" ] && grep -q "FINAL TABLE" "$log"; then echo "skip $log"; return 0; fi
  $PY centralized.main --experiment generate_results --dataset "$ds" --imputation_type zero --fusion_type "$fusion" \
      --extractor_nonlinear --backbone_depth "$depth" --missing_rate_single 0.5 --seeds $SEEDS | tee "$log"
}
for ds in $DATASETS; do for depth in 0 1 2 3; do for fusion in early mid late; do
  step "depth $ds d$depth $fusion" depth_run "$ds" "$depth" "$fusion"
done; done; done

echo
stamp "finished"
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED STAGES:"; printf '  %s\n' "${FAILED[@]}"; else echo "all stages OK"; fi
