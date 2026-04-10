# QLSTM Submission Package

This repository has been trimmed for submission around the TSLib benchmark workflow and the quantum wrapper models only.

The retained benchmark path is:

1. Run Task 1A TSLib forecasting benchmarks on the wildfire-risk panel.
2. Use `QCrossformer` Task 1A predictions to build chained Task 2 insurance datasets.
3. Run TSLib forecasting benchmarks on the chained Task 2 panels.
4. Render publication-style LaTeX/Markdown tables from the benchmark CSV outputs.

## Kept Structure

```text
.
├── data/
│   └── preprocessed/
│       ├── task1a/
│       └── task2/
├── libs/
│   └── Time-Series-Library/
│       ├── layers/
│       │   └── Quantum_Hybrid.py
│       ├── models/
│       │   ├── <TSLib benchmark models>.py
│       │   └── Q<model>.py
│       └── utils/
├── output/
│   ├── task1a_tslib_benchmarks/
│   └── task2_tslib_benchmarks_qcrossformer/
├── scripts/
│   ├── preprocess.py
│   ├── eval_tslib_task1a.py
│   ├── build_task2_updated_dataset.py
│   └── render_tslib_benchmark_table.py
├── requirements.txt
└── README.md
```

## Quantum Wrapper Integration

The shared hybrid block is implemented in:

- `libs/Time-Series-Library/layers/Quantum_Hybrid.py`

The benchmarked quantum wrappers are the `Q*.py` files in:

- `libs/Time-Series-Library/models/`

Each wrapper keeps the upstream TSLib model intact and applies the shared `QuantumResidualBlock` to the model forecast output.

## Environment

Create an environment and install:

```bash
pip install -r requirements.txt
```

The submission keeps the vendor copy of `Time-Series-Library`, so no extra package install is needed for the local model code itself.

## Preprocessing

The task-oriented raw-data preprocessing utility is also kept:

```bash
python scripts/preprocess.py
python scripts/preprocess.py --task task1a
python scripts/preprocess.py --task task2
python scripts/preprocess.py --no-extended
```

This script rebuilds the retained benchmark inputs from the raw joined dataset:

- `data/preprocessed/task1a/wildfire_risk_minimal.csv`
- `data/preprocessed/task1a/wildfire_risk_extended.csv`
- `data/preprocessed/task2/insurance_minimal.csv`
- `data/preprocessed/task2/insurance_extended.csv`

## Task 1A Benchmark

Classical TSLib models:

```bash
python scripts/eval_tslib_task1a.py \
  --datasets minimal extended \
  --output-dir output/task1a_tslib_benchmarks
```

Quantum wrappers only:

```bash
python scripts/eval_tslib_task1a.py \
  --datasets minimal extended \
  --quantum-only \
  --output-dir output/task1a_tslib_benchmarks
```

Include both classical and quantum variants:

```bash
python scripts/eval_tslib_task1a.py \
  --datasets minimal extended \
  --include-quantum \
  --output-dir output/task1a_tslib_benchmarks
```

Render the Task 1A table:

```bash
python scripts/render_tslib_benchmark_table.py \
  output/task1a_tslib_benchmarks/results.csv \
  --label tab:task1a-tslib
```

## Task 2 Chained Dataset

Build the chained Task 2 datasets from Task 1A `QCrossformer` predictions:

```bash
python scripts/build_task2_updated_dataset.py
```

This writes:

- `data/preprocessed/task2/insurance_minimal_qcrossformer.csv`
- `data/preprocessed/task2/insurance_extended_qcrossformer.csv`

The script adds a per-ZIP chained feature:

- `task1a_qcrossformer_risk_2021`

and a numeric provenance flag:

- `task1a_qcrossformer_risk_is_predicted`

## Task 2 Benchmark

Run the chained Task 2 benchmark:

```bash
python scripts/eval_tslib_task1a.py \
  --datasets \
    data/preprocessed/task2/insurance_minimal_qcrossformer.csv \
    data/preprocessed/task2/insurance_extended_qcrossformer.csv \
  --target earned_premium \
  --models all \
  --output-dir output/task2_tslib_benchmarks_qcrossformer
```

Render the Task 2 table:

```bash
python scripts/render_tslib_benchmark_table.py \
  output/task2_tslib_benchmarks_qcrossformer/results.csv \
  --caption "TSLib benchmark on Task 2 insurance-premium forecasting with the QCrossformer-chained wildfire-risk feature. Models are sorted by average RMSE rank across the minimal and extended chained dataset variants. Best values are bolded and second-best values are underlined within each dataset/metric block." \
  --label tab:task2-tslib-qcrossformer
```

This writes:

- `output/task2_tslib_benchmarks_qcrossformer/results_paper_table.tex`
- `output/task2_tslib_benchmarks_qcrossformer/results_paper_table.md`
- `output/task2_tslib_benchmarks_qcrossformer/results_paper_table.csv`

## Included Results

Task 1A benchmark artifacts are under:

- `output/task1a_tslib_benchmarks/`

Task 2 chained benchmark artifacts are under:

- `output/task2_tslib_benchmarks_qcrossformer/`

The current Task 2 best models by `R^2` in the retained results are:

- Minimal chained dataset: `Informer` (`R^2 = 0.7881`)
- Extended chained dataset: `Informer` (`R^2 = 0.7694`)

## License

MIT. The vendored `Time-Series-Library` subtree keeps its upstream license files.
