# QLSTM Benchmark Workflow

This repository is the trimmed benchmark package for the wildfire-risk and insurance-premium challenge workflow.

`challenge.md` keeps the original challenge statement. The retained data in this repo is narrower: both Task 1A and Task 2 are built from complete 2018-2021 ZIP/year panels, so the current scripts use 2018-2020 as history and predict 2021. The default runner now uses a deployment-style workflow: hold out a validation subset of ZIP panels for model selection, refit on all ZIP panels, and emit predictions for the full 2021 panel. That is the main sanity-check caveat to keep in mind when comparing the codebase to the original 2023 challenge wording.

## What Is In Scope

- [`scripts/preprocess.py`](/home/tawheed/exp/project_qml/QLSTM/scripts/preprocess.py): rebuild the retained Task 1A and Task 2 CSVs from the raw joined dataset.
- [`scripts/task1a.py`](/home/tawheed/exp/project_qml/QLSTM/scripts/task1a.py): benchmark classical and quantum TSLib forecasting models on either Task 1A or Task 2 style CSVs.
- [`scripts/task2.py`](/home/tawheed/exp/project_qml/QLSTM/scripts/task2.py): build chained Task 2 datasets from Task 1A prediction outputs.
- [`libs/Time-Series-Library/layers/Quantum_Hybrid.py`](/home/tawheed/exp/project_qml/QLSTM/libs/Time-Series-Library/layers/Quantum_Hybrid.py): shared `QuantumResidualBlock` used by the `Q*` wrapper models in [`libs/Time-Series-Library/models`](/home/tawheed/exp/project_qml/QLSTM/libs/Time-Series-Library/models).
- [`libs/Time-Series-Library/models/QFreTS_Deep.py`](/home/tawheed/exp/project_qml/QLSTM/libs/Time-Series-Library/models/QFreTS_Deep.py): deep quantum-enhanced FreTS with quantum channel gating, embedding re-uploading, and quantum latent prediction head.
- [`libs/Time-Series-Library/models/QCrossformer_Deep.py`](/home/tawheed/exp/project_qml/QLSTM/libs/Time-Series-Library/models/QCrossformer_Deep.py): deep quantum-enhanced Crossformer with quantum cross-dimension attention and encoder-decoder bridge.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preprocess The Retained Datasets

```bash
python scripts/preprocess.py
python scripts/preprocess.py --task task1a
python scripts/preprocess.py --task task2
python scripts/preprocess.py --no-extended
```

Main outputs:

- `data/preprocessed/task1a/wildfire_risk_minimal.csv`
- `data/preprocessed/task1a/wildfire_risk_extended.csv`
- `data/preprocessed/task2/insurance_minimal.csv`
- `data/preprocessed/task2/insurance_extended.csv`

## Task 1A Benchmarks

Default evaluation mode:

- `--split-mode full_2021` is now the default.
- The runner uses train/validation ZIP splits only for epoch selection.
- After selecting the best epoch, it refits on all ZIP panels and predicts the full 2021 panel.
- The legacy held-out ZIP benchmark is still available through `--split-mode zip_holdout`.

Classical models only:

```bash
python scripts/task1a.py \
  --datasets minimal extended \
  --models all \
  --output-dir output/task1a_classical \
  --run-tag task1a_classical_baseline
```

Classical Models and Quantum wrappers only:

```bash
# Run just the deep quantum models with full-2021 prediction output
python scripts/task1a.py --models QFreTS_Deep QCrossformer_Deep \
  --n-qubits 8 \
  --n-qlayers 2 \
  --data-reupload \
  --output-dir output/task1a_dq_nqb8_nl2 \
  --run-tag task1a_deep_quantum_full2021

# Run all quantum models (includes shallow Q* wrappers + deep versions)
python scripts/task1a.py --quantum-only --n-qubits 4
```


### Task 1A Output Layout

Every run now writes to a dedicated experiment directory:

```text
output/task1a_tslib_benchmarks/
├── latest_run.txt
├── results.csv
├── skipped.csv
└── runs/<run-tag>/
    ├── results.csv
    ├── skipped.csv
    ├── run_config.json
    ├── predictions/
    └── histories/
```

Notes:

- `runs/<run-tag>/...` is the immutable experiment record for that run.
- `<output-dir>/results.csv` and `skipped.csv` are the latest snapshot for convenience.
- `run_config.json` stores the benchmark settings, including `split_mode`, `n_qubits`, `n_qlayers`, backend, splits, and seed.
- `--nqbits`, `--n-layers`, and `--nlayers` are accepted as aliases for the quantum sweep arguments.

## Deep Quantum Hybrid Models

In addition to the shallow `Q*` wrappers (which append a quantum block to the classical model output), two **deep quantum hybrid** models integrate VQC circuits inside the architecture itself:

### QFreTS_Deep

Quantum circuits replace or augment three internal stages of FreTS:

| Stage | Replaces | Mechanism |
|-------|----------|-----------|
| Quantum Channel Gate | `MLP_channel` | Global pool → VQC → sigmoid → channel modulation. Qubit entanglement captures cross-variable correlations. |
| Quantum Embed Refine | N/A (added) | After temporal FFT, time-pooled embeddings pass through a second VQC (data re-uploading across layers). |
| Quantum Latent Head | FC layer | `FC → VQC → FC` in the prediction pathway. Quantum bottleneck in latent space before final projection. |

### QCrossformer_Deep

Quantum circuits replace two key computation points in Crossformer:

| Stage | Replaces | Mechanism |
|-------|----------|-----------|
| Quantum Cross-Dim Attention | Router-based `dim_sender`/`dim_receiver` in `TwoStageAttentionLayer` | VQC processes variable representations; entanglement drives cross-variable mixing instead of learned routers. Applied in every encoder layer. |
| Quantum Encoder Bridge | Direct encoder→decoder pass | Last encoder output passes through a VQC before the decoder, creating a quantum bottleneck over the most abstract representations. |

### Running Deep Quantum Models

```bash
# Run only the deep quantum models
python scripts/task1a.py \
  --datasets minimal extended \
  --models QFreTS_Deep QCrossformer_Deep \
  --n-qubits 4 \
  --n-qlayers 2 \
  --data-reupload \
  --output-dir output/task1a_deep_quantum \
  --run-tag deep_q_nq4_ql2_full2021

# Compare shallow vs deep quantum FreTS
python scripts/task1a.py \
  --datasets minimal extended \
  --models QFreTS QFreTS_Deep \
  --n-qubits 4 \
  --n-qlayers 1 \
  --output-dir output/task1a_shallow_vs_deep \
  --run-tag shallow_vs_deep_frets
```

## Quantum Sweeps

Example sweep over qubit count and variational depth:

```bash
for nq in 2 4 6; do
  for ql in 1 2 3; do
    python scripts/task1a.py \
      --datasets minimal extended \
      --models QCrossformer QInformer QTransformer \
      --n-qubits "$nq" \
      --n-qlayers "$ql" \
      --output-dir output/task1a_tslib_benchmarks \
      --run-tag "sweep_nq${nq}_ql${ql}_full2021"
  done
done
```

Single-qubit runs are supported. The quantum residual block now skips entangling gates automatically when `n_qubits=1`.

## Build Chained Task 2 Datasets

Use the latest compatible Task 1A run automatically:

```bash
python scripts/task2.py \
  --datasets minimal extended \
  --task1a-model QCrossformer
```

Use a specific Task 1A run:

```bash
python scripts/task2.py \
  --datasets minimal extended \
  --task1a-model QCrossformer \
  --run-tag task1a_quantum_nq4_ql2
```

Use a different Task 1A model and explicit suffix:

```bash
python scripts/task2.py \
  --datasets minimal extended \
  --task1a-model QInformer \
  --run-tag sweep_nq4_ql2 \
  --suffix qinformer_chain_nq4_ql2
```

The default output suffix is derived from the Task 1A model and run tag, so chained datasets from different Task 1A runs do not overwrite each other.

Recommended Task 2 workflow from the deep-quantum Task 1A run:

1. Build the chained Task 2 datasets from the Task 1A run root:

```bash
python scripts/task2.py \
  --datasets minimal extended \
  --task1a-model QCrossformer_Deep \
  --prediction-root /home/tawheed/exp/project_qml/QLSTM/output/task1a_dq_nqb4_nl1/runs/task1a-deep-quantum-full2021 \
  --output-dir data/preprocessed/task2
```

2. Launch the downstream Task 2 benchmark immediately after chaining:

```bash
python scripts/task2.py \
  --datasets minimal extended \
  --task1a-model QCrossformer_Deep \
  --prediction-root /home/tawheed/exp/project_qml/QLSTM/output/task1a_dq_nqb4_nl1/runs/task1a-deep-quantum-full2021 \
  --output-dir data/preprocessed/task2 \
  --run-benchmarks \
  --benchmark-models all \
  --benchmark-output-dir output/task2_tslib_benchmarks \
  --benchmark-run-tag task2_qcrossformer_deep_chain_full2021
```

3. Include quantum wrappers in the downstream Task 2 benchmark:

```bash
python scripts/task2.py \
  --datasets minimal extended \
  --task1a-model QCrossformer_Deep \
  --prediction-root /home/tawheed/exp/project_qml/QLSTM/output/task1a_dq_nqb4_nl1/runs/task1a-deep-quantum-full2021 \
  --output-dir data/preprocessed/task2 \
  --run-benchmarks \
  --benchmark-models all \
  --benchmark-include-quantum \
  --benchmark-device cuda \
  --benchmark-n-qubits 4 \
  --benchmark-n-qlayers 1 \
  --benchmark-data-reupload \
  --benchmark-output-dir output/task2_tslib_benchmarks \
  --benchmark-run-tag task2_qcrossformer_deep_chain_full2021_gpu
```

4. Use a GPU PennyLane backend only if it is installed and working in the current environment:

```bash
python scripts/task2.py \
  --datasets minimal extended \
  --task1a-model QCrossformer_Deep \
  --prediction-root /home/tawheed/exp/project_qml/QLSTM/output/task1a_dq_nqb4_nl1/runs/task1a-deep-quantum-full2021 \
  --output-dir data/preprocessed/task2 \
  --run-benchmarks \
  --benchmark-models all \
  --benchmark-include-quantum \
  --benchmark-device cuda \
  --benchmark-quantum-backend lightning.gpu \
  --benchmark-n-qubits 4 \
  --benchmark-n-qlayers 1 \
  --benchmark-data-reupload \
  --benchmark-output-dir output/task2_tslib_benchmarks \
  --benchmark-run-tag task2_qcrossformer_deep_chain_full2021_lightning_gpu
```

Notes:

- Prefer pointing `--prediction-root` at the Task 1A run root. `task2.py` resolves the `predictions/` directory automatically.
- If you point `--prediction-root` directly at a `predictions/` directory, `task2.py` infers the parent run tag and preserves it in the chained dataset suffix.
- When the supplied Task 1A predictions cover the full 2021 panel, all ZIPs receive model-predicted 2021 wildfire risk values in the chained feature.
- `--run-benchmarks` forwards the generated chained CSV paths into `scripts/task1a.py` with `--target earned_premium`.
- Use `--benchmark-device cuda` to force the downstream TSLib benchmark onto the GPU when CUDA is available.
- For quantum wrappers, you can also forward a GPU-capable PennyLane backend with `--benchmark-quantum-backend lightning.gpu` if that backend is installed.
- If rerunning the chaining step into the same output directory fails because the chained feature already exists, either delete the generated Task 2 CSVs or pass a new `--suffix`.
- If you see a generic CUDA or `cuda unknown error`, first retry without `--benchmark-quantum-backend lightning.gpu`. That error usually indicates a CUDA runtime / driver / PennyLane backend mismatch rather than a `task2.py` path issue.
- A safe debugging sequence is:
  1. `--benchmark-device cuda` with classical models only.
  2. `--benchmark-device cuda --benchmark-include-quantum` while keeping the default quantum backend.
  3. Add `--benchmark-quantum-backend lightning.gpu` only after the first two steps work.
- If you want to validate model compatibility before a long run, add `--benchmark-preflight-only`.

## Task 2 Benchmarks

After building chained Task 2 CSVs, benchmark them with the same runner:

```bash
python scripts/task1a.py \
  --datasets \
    data/preprocessed/task2/insurance_minimal_qcrossformer_task1a_quantum_nq4_ql2.csv \
    data/preprocessed/task2/insurance_extended_qcrossformer_task1a_quantum_nq4_ql2.csv \
  --target earned_premium \
  --models all \
  --output-dir output/task2_tslib_benchmarks \
  --run-tag task2_classical_qcrossformer_chain
```

The Task 2 benchmark inherits the same default `full_2021` split mode. Use `--benchmark-split-mode zip_holdout` from `task2.py`, or `--split-mode zip_holdout` when calling `task1a.py` directly, only if you explicitly want the older held-out ZIP evaluation.

Include quantum wrappers in the Task 2 benchmark:

```bash
python scripts/task1a.py \
  --datasets \
    data/preprocessed/task2/insurance_minimal_qcrossformer_task1a_quantum_nq4_ql2.csv \
    data/preprocessed/task2/insurance_extended_qcrossformer_task1a_quantum_nq4_ql2.csv \
  --target earned_premium \
  --models all \
  --include-quantum \
  --n-qubits 4 \
  --n-qlayers 2 \
  --output-dir output/task2_tslib_benchmarks \
  --run-tag task2_full_qcrossformer_chain
```

## Prediction Visualizations

Use [`scripts/plot_regression_predictions.py`](/home/tawheed/exp/project_qml/QLSTM/scripts/plot_regression_predictions.py) to create a paper-ready two-panel predicted-vs-actual figure for Task 1A and Task 2.

Auto-select the best available prediction CSV from the Task 1A and Task 2 `results.csv` files:

```bash
python scripts/plot_regression_predictions.py \
  --output output/figures/predicted_vs_actual_task1a_task2
```

Select specific Task 1A and Task 2 rows from the results snapshots:

```bash
python scripts/plot_regression_predictions.py \
  --task1a-results output/task1a_dq_nqb4_nl1/results.csv \
  --task1a-model QCrossformer_Deep \
  --task1a-dataset extended \
  --task2-results output/task2_tslib_benchmarks/results.csv \
  --task2-model Nonstationary_Transformer \
  --task2-dataset insurance_extended_qcrossformer_deep_task1a_deep_quantum_full2021 \
  --output output/figures/predicted_vs_actual_paper
```

Use direct prediction CSVs instead of results snapshots:

```bash
python scripts/plot_regression_predictions.py \
  --task1a-prediction output/task1a_dq_nqb4_nl1/runs/task1a-deep-quantum-full2021/predictions/extended__QCrossformer_Deep.csv \
  --task2-prediction output/task2_tslib_benchmarks/runs/task2-qcrossformer-deep-chain-full2021/predictions/insurance_extended_qcrossformer_deep_task1a_deep_quantum_full2021__Nonstationary_Transformer.csv \
  --output output/figures/predicted_vs_actual_direct
```

Notes:

- The script writes both `.png` and `.pdf` outputs using the same stem.
- When driven from `results.csv`, it can automatically select the best row by `r2`, `rmse`, `mse`, or `mae` via `--selection-metric`.
- The figure uses a parity line, fitted regression line, hexbin density, and an inset metrics box for each task.

## Suggested Benchmarks

- Classical Task 1A baseline: `--models all`
- Quantum-only Task 1A comparison: `--quantum-only`
- Classical vs quantum Task 1A parity run: `--include-quantum`
- **Shallow vs deep quantum**: `--models QFreTS QFreTS_Deep QCrossformer QCrossformer_Deep`
- **Deep quantum sweep**: vary `--n-qubits`, `--n-qlayers`, `--data-reupload` across `QFreTS_Deep QCrossformer_Deep`
- Task 1A hyperparameter sweeps: vary `--n-qubits`, `--n-qlayers`, `--n-esteps`, `--data-reupload`, and seed
- Task 2 chained-data ablations: swap `--task1a-model` across `QCrossformer`, `QInformer`, `QTransformer`, or other saved Task 1A predictors
- Task 2 classical vs quantum comparison: run the chained Task 2 datasets with `--models all` and then `--include-quantum`

## Included Outputs

The repository still contains retained benchmark artifacts under:

- [`output/task1a_tslib_benchmarks`](/home/tawheed/exp/project_qml/QLSTM/output/task1a_tslib_benchmarks)
- [`output/task2_tslib_benchmarks_qcrossformer`](/home/tawheed/exp/project_qml/QLSTM/output/task2_tslib_benchmarks_qcrossformer)

Treat those as snapshots. New runs should be written through the scripts above so the run-tagged experiment layout is preserved.

## License

MIT. The vendored `Time-Series-Library` subtree keeps its upstream license files.
