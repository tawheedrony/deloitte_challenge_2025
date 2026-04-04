# Quantum LSTM — 2026 Quantum Sustainability Challenge

![CLSTM drawio](https://github.com/user-attachments/assets/8ef76e0c-c6de-4a5b-9c9d-4111b9c0b2e9)

A hybrid quantum-classical time series framework for the **2026 Quantum Sustainability Challenge** — predicting California wildfire risk and insurance premiums from historical ZIP-level data.

---

## Table of Contents
- [Challenge Overview](#challenge-overview)
- [Task 1A — Wildfire Risk Prediction](#task-1a--wildfire-risk-prediction)
- [Task 1B — Quantum vs Classical Evaluation](#task-1b--quantum-vs-classical-evaluation)
- [Task 2 — Insurance Premium Prediction](#task-2--insurance-premium-prediction)
- [Models](#models)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Quickstart](#quickstart)
- [Configs](#configs)
- [NN Benchmark Results](#nn-benchmark-results)
- [Logging](#logging)
- [License](#license)

---

## Challenge Overview

The competition presents two tasks using wildfire and insurance datasets for California ZIP codes:

| Task | Goal | Target | Data |
|------|------|--------|------|
| **1A** | Predict wildfire risk using a quantum/hybrid QML model | `avg_fire_risk_score` (0–4) or `fire_occurred` (binary) | 2018–2020 → predict 2021 |
| **1B** | Evaluate quantum vs classical performance | Comparison report | Same as 1A |
| **2** | Predict insurance premiums using time series | `earned_premium` ($) | 2018–2020 → predict 2021 |

---

## Task 1A — Wildfire Risk Prediction

A single script trains **all** model architectures on wildfire risk prediction, evaluates them, and saves outputs.

Quantum models are trained at **4, 8, and 12 qubits**. Classical baselines are trained once. Optionally tests **data re-uploading** for cQLSTM and QSSM.

```bash
# prerequisite
python scripts/preprocess.py --engineered

# train + eval all models at 4, 8, 12 qubits
python scripts/task1a.py

# specific models / qubit counts
python scripts/task1a.py --models cQLSTM LSTM --qubits 4 8
python scripts/task1a.py --models cQLSTM QSSM --reupload   # test data re-uploading
python scripts/task1a.py --epochs 20                        # quick run
```

**Output** saved to `output/task1a/`:
- `predictions/` — per-model CSV with predicted vs actual risk scores
- `task1a_results.csv` — comparison table across all experiments
- `task1a_summary.txt` — full report with best quantum vs classical comparison

### Resource Requirements

```bash
python scripts/resource_report.py                     # 4-qubit report
python scripts/resource_report.py --n_qubits 8        # 8-qubit report
python scripts/resource_report.py --n_qubits 12       # 12-qubit report
```

Reports circuit depth, gate counts (1q/2q), parameter counts, VQC structure, and timing benchmarks.

### Quantum Improvements

- **Data re-uploading**: re-encodes classical data between variational layers, increasing expressibility. Enabled with `--reupload` flag or `data_reupload: true` in config.
- **Multi-qubit scaling**: 4 → 8 → 12 qubits to study the effect of Hilbert space dimension on predictive power.

---

## Task 1B — Quantum vs Classical Evaluation

Task 1B evaluation is built into `task1a.py` — the summary report automatically includes quantum vs classical comparison with parameter efficiency and R² delta.

For a standalone comparison report:

```bash
python scripts/task1b_report.py --run "lightning_logs/task1a_*/version_0" \
    --out output/task1b_report.txt
```

The report covers:
- **Performance comparison** — R²/RMSE/MAE ranked table
- **Parameter efficiency** — quantum models achieve competitive results with 10–30x fewer parameters
- **Computational cost** — forward-pass timing benchmarks
- **Advantages/disadvantages** — qualitative analysis

---

## Task 2 — Insurance Premium Prediction

Classical time series and ML models predict 2021 earned insurance premiums.

### Classical ML Models

```bash
# all models (XGBoost, LightGBM, RandomForest, GradientBoosting, ARIMA, Prophet)
python scripts/task2.py

# specific models
python scripts/task2.py --models xgboost lgbm rf
python scripts/task2.py --models arima prophet

# use engineered features (20+ features)
python scripts/task2.py --dataset engineered

# also train NN models (LSTM, cQLSTM) for comparison
python scripts/task2.py --include-nn
```

**Available models:**

| Model | Type | Library |
|-------|------|---------|
| XGBoost | Gradient boosted trees | `xgboost` |
| LightGBM | Gradient boosted trees | `lightgbm` |
| RandomForest | Ensemble bagging | `sklearn` |
| GradientBoosting | Gradient boosted trees | `sklearn` |
| ARIMA | Per-ZIP univariate TS | `statsmodels` |
| Prophet | Per-ZIP univariate TS | `prophet` |

**Output** saved to `output/task2/`:
- `predictions/` — per-model CSV with predicted vs actual premiums
- `task2_results.csv` — comparison table
- `task2_summary.txt` — full report

### Neural Network Models (Task 2)

NN models can also be trained separately on premium prediction:

```bash
python scripts/train.py --config configs/model/cQLSTM.yaml
python scripts/train.py --config configs/model/LSTM.yaml

# evaluate
python scripts/eval.py --run "lightning_logs/*/version_0"
```

---

## Models

### Neural Network Architectures

| Model | Type | Gate structure | VQC | Params (hidden=64) |
|---|---|---|---|---|
| **LSTM** | Classical | f, i, g, o | — | 18,944 |
| **cLSTM** | Classical + decay | f, i, g, o + decay | — | 18,953 |
| **QLSTM** | Quantum | f, i, g, o (one VQC each) | 4 × 4-qubit | 616 |
| **cQLSTM** | Quantum + decay | f, i, g, o + decay (unified VQC) | 1 × 4-qubit | 1,597 |
| **SSM** | Classical SSM | g, delta | — | 9,481 |
| **QSSM** | Quantum SSM | g, delta (VQC) | 1 × 4-qubit | 957 |
| **cQSSM** | Quantum SSM + decay | g, i, delta + decay | 1 × 4-qubit | 1,435 |
| **DeltaNet** | Classical delta-rule | Q, K, V | — | 7,391 |
| **QDeltaNet** | Quantum delta-rule | Q, K, V (3 VQCs) | 3 × 4-qubit | 2,115 |

### VQC Design

```
Encoding per qubit i:  H → RY(arctan(x_i)) → RZ(arctan(x_i²))
Ansatz (n_qlayers):    CNOT ring → RX/RY/RZ variational rotations
Data re-upload (opt):  Re-encode RY/RZ between variational layers
```

### Classical ML Models (Task 2)

XGBoost, LightGBM, RandomForest, GradientBoosting, ARIMA, Prophet — see [Task 2](#task-2--insurance-premium-prediction).

---

## Project Structure

```
QLSTM/
├── scripts/
│   ├── train.py                # unified NN training entry point
│   ├── eval.py                 # NN evaluation & multi-run comparison
│   ├── preprocess.py           # raw data → preprocessed/engineered CSV
│   ├── visualize.py            # general predicted-vs-actual plots
│   │
│   ├── task1a.py               # Task 1A: train + eval ALL models (4/8/12q)
│   ├── task1a_visualize.py     # Task 1A: risk score visualization
│   ├── resource_report.py      # Task 1A: quantum resource requirements
│   ├── task1b_report.py        # Task 1B: quantum vs classical report
│   │
│   ├── task2.py                # Task 2: ARIMA, Prophet, XGBoost, LightGBM, RF, GBR
│   ├── task2_visualize.py      # Task 2: premium prediction visualization
│   │
│   └── eda.py                  # exploratory data analysis
│
├── configs/
│   ├── model/                  # one YAML per NN model (+ Task 1A variants)
│   └── dataset/                # dataset configs (preprocessed, engineered, wildfire_risk, wildfire_fire)
│
├── src/
│   ├── models/                 # LSTM, cLSTM, QLSTM, cQLSTM, SSM, QSSM, cQSSM, DeltaNet, QDeltaNet
│   └── modules/                # Lightning wrapper, data, scheduler, callbacks, utils
│
├── data/                       # raw + preprocessed CSVs
├── output/                     # task1a/ and task2/ results, predictions, reports
├── lightning_logs/             # NN training runs (checkpoints + TensorBoard)
└── challenge.md                # competition description & todos
```

---

## Requirements

Python >= 3.10. Create a dedicated environment:

```bash
conda create -n qlstm python=3.11 -y
conda activate qlstm
pip install -r requirements.txt

# for Task 2 classical models
pip install xgboost lightgbm prophet statsmodels
```

---

## Dataset

Raw data lives in `data/`. The preprocessing script aggregates to one row per (ZIP, Year):

```bash
python scripts/preprocess.py               # base: 9 features
python scripts/preprocess.py --engineered  # extended: 20+ features
```

**Task 1A features** (15 engineered): fire risk, weather, fire history, risk composition, socioeconomic, temporal

**Task 2 features** (9 base): fire risk, exposure, losses, population, income, rolling premium, temporal

---

## Quickstart

```bash
# 1. preprocess
python scripts/preprocess.py
python scripts/preprocess.py --engineered

# 2. Task 1A: train all models at 4/8/12 qubits
python scripts/task1a.py

# 3. Task 1B: comparison report (included in task1a.py output)
python scripts/task1b_report.py --run "lightning_logs/task1a_*/version_0"

# 4. Task 2: classical ML models for premium prediction
python scripts/task2.py

# 5. Resource report
python scripts/resource_report.py --out output/resource_report.txt
```

---

## Configs

Each NN model has its own YAML in `configs/model/`. Override any key via CLI:

```bash
python scripts/train.py --config configs/model/cQLSTM_task1a.yaml --model.n_qubits 8
python scripts/train.py --config configs/model/cQLSTM.yaml --model.data_reupload true
```

---

## NN Benchmark Results

Premium prediction (Task 2) — neural network models:

```
══════════════════════════════════════════════════════════════════════
  COMPARISON SUMMARY  (ranked by Test R²)
══════════════════════════════════════════════════════════════════════
                 Run  Epochs  Best val/loss  Test R²  RMSE ($)  MAE ($)  MdAPE (%)
Model
cQLSTM     version_0      40       0.030899   0.8317   2051810   753291       34.3
DeltaNet   version_0      32       0.026857   0.8172   2138799   796730       33.6
QSSM       version_0      40       0.027649   0.8131   2162429   735898       38.7
cLSTM      version_0      35       0.025572   0.8094   2183685   749078       34.1
QDeltaNet  version_0      37       0.026081   0.8092   2184814   806409       45.9
LSTM       version_0      40       0.029618   0.8023   2224103   786203       36.4
SSM        version_0      40       0.024315   0.7955   2262139   730289       30.7
cQSSM      version_0      36       0.072250   0.6946   2764082  1063362       37.9
QLSTM      version_0      40       0.035170  -1.2738   7542416  7199800     1120.2
```

## Logging

```bash
tensorboard --logdir lightning_logs --bind_all
```

---

## License

MIT License — see [LICENSE](LICENSE).
