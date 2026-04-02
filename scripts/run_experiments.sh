#!/usr/bin/env bash
# run_experiments.sh
# ==================
# Trains every model x dataset x seed combination, then runs eval.py
# across all completed runs for a side-by-side comparison.
#
# Usage:
#   bash scripts/run_experiments.sh              # all models, both datasets, 10 seeds
#   MODELS="LSTM cLSTM" bash scripts/run_experiments.sh   # subset of models
#
# Output layout:
#   lightning_logs/{MODEL}_{DATASET}_seed{SEED}/version_0/
#
# Eval summary written to:
#   output/eval_summary_{TIMESTAMP}.txt

set -euo pipefail

# ── configurable ──────────────────────────────────────────────────────────
MODELS="${MODELS:-LSTM cLSTM QLSTM cQLSTM SSM QSSM cQSSM DeltaNet QDeltaNet}"
DATASETS="${DATASETS:-preprocessed engineered}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPTS_DIR")"
OUTPUT_DIR="$ROOT_DIR/output"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_DIR/logs_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ── preprocessing ─────────────────────────────────────────────────────────
echo "================================================================"
echo "  STEP 1 — Preprocessing"
echo "================================================================"

cd "$ROOT_DIR"

if [ ! -f "data/wildfire_preprocessed.csv" ]; then
    echo "  Running base preprocessing …"
    python scripts/preprocess.py 2>&1 | tee "$LOG_DIR/preprocess_base.log"
else
    echo "  data/wildfire_preprocessed.csv already exists, skipping."
fi

if [ ! -f "data/wildfire_engineered.csv" ]; then
    echo "  Running engineered preprocessing …"
    python scripts/preprocess.py --engineered 2>&1 | tee "$LOG_DIR/preprocess_engineered.log"
else
    echo "  data/wildfire_engineered.csv already exists, skipping."
fi

# ── training ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  STEP 2 — Training"
echo "================================================================"

FAILED_RUNS=()
COMPLETED_RUNS=()

for MODEL in $MODELS; do
    MODEL_CFG="configs/model/${MODEL}.yaml"
    if [ ! -f "$MODEL_CFG" ]; then
        echo "  [WARN] Config not found: $MODEL_CFG — skipping $MODEL"
        continue
    fi

    for DATASET in $DATASETS; do
        DATASET_CFG="configs/dataset/${DATASET}.yaml"
        if [ ! -f "$DATASET_CFG" ]; then
            echo "  [WARN] Dataset config not found: $DATASET_CFG — skipping"
            continue
        fi

        for SEED in $SEEDS; do
            RUN_NAME="${MODEL}_${DATASET}_seed${SEED}"
            RUN_LOG="$LOG_DIR/${RUN_NAME}.log"

            # skip if already completed
            if ls lightning_logs/"$RUN_NAME"/version_*/run_config.yaml 2>/dev/null | grep -q .; then
                echo "  [SKIP] $RUN_NAME — checkpoint already exists"
                # still collect for eval
                VERSION_DIR=$(ls -d lightning_logs/"$RUN_NAME"/version_* 2>/dev/null | sort -V | tail -1)
                [ -n "$VERSION_DIR" ] && COMPLETED_RUNS+=("$VERSION_DIR")
                continue
            fi

            echo ""
            echo "  ── $RUN_NAME ──────────────────────────────────────"
            echo "     Model: $MODEL | Dataset: $DATASET | Seed: $SEED"

            set +e
            python scripts/train.py \
                --config "$MODEL_CFG" \
                --dataset "$DATASET_CFG" \
                --training.seed "$SEED" \
                --logging.run_name "$RUN_NAME" \
                2>&1 | tee "$RUN_LOG"
            EXIT_CODE=$?
            set -e

            if [ $EXIT_CODE -ne 0 ]; then
                echo "  [FAIL] $RUN_NAME (exit $EXIT_CODE) — see $RUN_LOG"
                FAILED_RUNS+=("$RUN_NAME")
                continue
            fi

            # find the version dir just created
            VERSION_DIR=$(ls -d lightning_logs/"$RUN_NAME"/version_* 2>/dev/null | sort -V | tail -1)
            if [ -n "$VERSION_DIR" ]; then
                COMPLETED_RUNS+=("$VERSION_DIR")
                echo "  [OK]   $RUN_NAME  →  $VERSION_DIR"
            else
                echo "  [WARN] $RUN_NAME completed but no version dir found"
                FAILED_RUNS+=("$RUN_NAME")
            fi
        done
    done
done

# ── training summary ──────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Training summary"
echo "================================================================"
echo "  Completed : ${#COMPLETED_RUNS[@]}"
echo "  Failed    : ${#FAILED_RUNS[@]}"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "  Failed runs:"
    for r in "${FAILED_RUNS[@]}"; do echo "    - $r"; done
fi

# ── evaluation ────────────────────────────────────────────────────────────
if [ ${#COMPLETED_RUNS[@]} -eq 0 ]; then
    echo ""
    echo "  No completed runs to evaluate. Exiting."
    exit 1
fi

echo ""
echo "================================================================"
echo "  STEP 3 — Evaluation"
echo "================================================================"

EVAL_OUTPUT="$OUTPUT_DIR/eval_summary_${TIMESTAMP}.txt"

# build --run args for eval.py
EVAL_ARGS=()
for RUN_DIR in "${COMPLETED_RUNS[@]}"; do
    EVAL_ARGS+=(--run "$RUN_DIR")
done

echo "  Evaluating ${#COMPLETED_RUNS[@]} runs …"
echo "  Output → $EVAL_OUTPUT"
echo ""

python scripts/eval.py "${EVAL_ARGS[@]}" 2>&1 | tee "$EVAL_OUTPUT"

echo ""
echo "================================================================"
echo "  Done."
echo "  Eval summary : $EVAL_OUTPUT"
echo "  Training logs: $LOG_DIR/"
echo "================================================================"
