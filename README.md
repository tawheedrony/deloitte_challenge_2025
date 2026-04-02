# Quantum LSTM — Wildfire Insurance Premium Forecasting

![CLSTM drawio](https://github.com/user-attachments/assets/8ef76e0c-c6de-4a5b-9c9d-4111b9c0b2e9)

A hybrid quantum-classical time series framework applied to the **2026 Quantum Sustainability Challenge** — predicting California wildfire insurance premiums from historical ZIP-level data.

Six model variants are provided: classical baselines (LSTM, SSM), decay-modified classicals (cLSTM), and quantum hybrids (QLSTM, cQLSTM, QSSM) that embed Variational Quantum Circuits (VQCs) via PennyLane.

---

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Quickstart](#quickstart)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Configs](#configs)
- [Benchmark Results](#benchmark-results)
- [Logging](#logging)
- [Demo App](#demo-app)
- [License](#license)

---

## Introduction

This repository presents a hybrid AI framework that integrates Quantum Machine Learning (QML) and classical Deep Learning for time series regression. Variational Quantum Circuits (VQCs) are embedded inside recurrent cell computations, enabling compact parameter footprints while maintaining competitive accuracy.

The framework is applied to the **2026 Quantum Sustainability Challenge**: predicting 2021 earned insurance premiums for 1,906 California ZIP codes using 3 years of historical wildfire risk, exposure, loss, and census data (2018–2020).

---

## Models

| Model | Type | Gate structure | VQC | Params (hidden=64) |
|---|---|---|---|---|
| **LSTM** | Classical | f, i, g, o | — | 18,944 |
| **cLSTM** | Classical + decay | f, i, g, o + decay | — | 18,953 |
| **QLSTM** | Quantum | f, i, g, o (one VQC each) | 4 × 4-qubit | 616 |
| **cQLSTM** | Quantum + decay | f, i, g, o + decay (unified VQC) | 1 × 4-qubit | 1,597 |
| **SSM** | Classical SSM | g, delta | — | 9,481 |
| **QSSM** | Quantum SSM | g, delta (VQC) | 1 × 4-qubit | 957 |

All quantum models share the same VQC design:
```
Encoding per qubit i:  H → RY(arctan(x_i)) → RZ(arctan(x_i²))
Ansatz (n_qlayers):    CNOT ring → RX/RY/RZ variational rotations
```

SSM-family recurrence (SSM, QSSM):
```
h_t = (1 - g_t) * h_{t-1}  +  g_t * delta_t
g_t = clamp(sigmoid(g_raw), 0.05, 0.95)
```

LSTM-family recurrence (LSTM, cLSTM, QLSTM, cQLSTM):
```
c_t = f_t * c_{t-1}  +  i_t * g_t
h_t = o_t * tanh(c_t)
# cLSTM/cQLSTM additionally apply: f_t *= (1 - decay_rate)
```

---

## Project Structure

```
QLSTM/
├── scripts/
│   ├── train.py                # unified training entry point
│   ├── eval.py                 # evaluation & multi-run comparison
│   ├── preprocess.py           # raw data → wildfire_preprocessed.csv
│   └── visualize.py            # predicted vs actual plots for any set of runs
│
├── configs/                    # one YAML per model
│   ├── LSTM.yaml
│   ├── cLSTM.yaml
│   ├── QLSTM.yaml
│   ├── cQLSTM.yaml             (default)
│   ├── SSM.yaml
│   └── QSSM.yaml
│
├── src/
│   ├── models/
│   │   ├── LSTM.py
│   │   ├── cLSTM.py
│   │   ├── QLSTM.py
│   │   ├── cQLSTM.py
│   │   ├── SSM.py              (vanilla state-space model)
│   │   └── QSSM.py             (quantum state-space model)
│   └── modules/
│       ├── model.py            (Lightning wrapper)
│       ├── data.py             (data utilities)
│       ├── scheduler.py        (warmup + cosine LR)
│       ├── callback.py
│       └── utils.py
│
├── data/
│   ├── abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv   (raw)
│   └── wildfire_preprocessed.csv                                (generated)
│
├── lightning_logs/
│   └── {ModelName}_{YYYYMMDD_HHMMSS}/
│       └── version_0/
│           ├── best-epoch=XX-valloss=X.XXXX.ckpt
│           ├── run_config.yaml
│           └── events.out.tfevents.*
│
└── app/                        (Gradio demo)
```

---

## Requirements

Python >= 3.10. Create a dedicated environment:

```bash
conda create -n qlstm python=3.11 -y
conda activate qlstm
pip install -r requirements.txt
```

---

## Dataset

Raw data lives in `data/` (four CSV files from the challenge). The preprocessing script aggregates them to one row per (ZIP, Year):

```bash
python scripts/preprocess.py
# writes  data/wildfire_preprocessed.csv  (7,624 rows, 1,906 ZIP codes × 4 years)
```

**Input sequence** — 3 years (2018–2020) × 9 features per year:

| Feature | Description |
|---|---|
| `avg_fire_risk_score` | Wildfire hazard score (0–4) |
| `earned_exposure` | Insured units in the ZIP |
| `cat_fire_losses` | Catastrophic fire losses ($) |
| `noncat_fire_losses` | Non-catastrophic fire losses ($) |
| `total_population` | Census population |
| `median_income` | Median household income ($) |
| `premium_rolling_mean` | Expanding mean of prior premiums |
| `year_sin` / `year_cos` | Cyclical temporal encoding |

**Target** — `earned_premium` for 2021. Dataset split: 70% train / 15% val / 15% test.

---

## Quickstart

```bash
# 1. preprocess
python scripts/preprocess.py

# 2. train with default model (cQLSTM)
python scripts/train.py

# 3. evaluate
python scripts/eval.py --run lightning_logs/cQLSTM_YYYYMMDD_HHMMSS/version_0

# 4. visualize
python scripts/visualize.py --run "lightning_logs/*/version_0" --out results.png
```

---

## Training

```bash
# use a model-specific config
python scripts/train.py --config configs/LSTM.yaml
python scripts/train.py --config configs/cQLSTM.yaml
python scripts/train.py --config configs/QSSM.yaml

# override any config key via dot notation
python scripts/train.py --config configs/cQLSTM.yaml --training.max_epochs 60
python scripts/train.py --config configs/LSTM.yaml --model.hidden_size 128 --training.lr 1e-3

# custom run name (controls the lightning_logs subfolder)
python scripts/train.py --logging.run_name my_experiment
```

Each run saves to `lightning_logs/{run_name}/version_0/` and writes a `run_config.yaml` alongside the checkpoint so the eval script can fully reconstruct the run.

---

## Evaluation

```bash
# single run — full report with learning curves + dollar-scale metrics
python scripts/eval.py --run lightning_logs/cQLSTM_20260323_135146/version_0

# compare multiple runs side by side
python scripts/eval.py \
    --run lightning_logs/LSTM_*/version_0 \
    --run lightning_logs/cQLSTM_*/version_0 \
    --run lightning_logs/SSM_*/version_0

# compare all runs at once (glob)
python scripts/eval.py --run "lightning_logs/*/version_0"
```

Output includes:
- Learning curve summary (ep1 → final → best for train/val loss and R²)
- Test set metrics in scaled and original dollar space (R², RMSE, MAE, MdAPE)
- Multi-run ranked comparison table when more than one run is passed

---

## Visualization

```bash
# all runs — display interactively
python scripts/visualize.py --run "lightning_logs/*/version_0"

# specific models — save to file
python scripts/visualize.py \
    --run lightning_logs/LSTM_*/version_0 \
    --run lightning_logs/cQLSTM_*/version_0 \
    --out comparison.png
```

Produces a 3-panel figure:
- **Scatter** — predicted vs actual premium per model (with identity line)
- **RMSE bar** — test RMSE per model in dollar scale
- **R² bar** — test R² per model

---

## Configs

Each model has its own YAML in `configs/`. All share the same `data` and `training` blocks; only the `model` section differs.

```yaml
# configs/cQLSTM.yaml
model:
  name: cQLSTM
  hidden_size: 64
  n_qubits: 4
  n_qlayers: 1
  n_esteps: 1
  decay_rate: 0.1

training:
  seed: 42
  batch_size: 32
  max_epochs: 40
  lr: 2.0e-3
  weight_decay: 1.0e-4
  warmup_epochs: 3
  warmup_start_factor: 0.01
  early_stop_patience: 10

logging:
  log_dir: lightning_logs
  run_name: null          # null → auto "{model.name}_{YYYYMMDD_HHMMSS}"
```

---

## Benchmark Results

python scripts/eval.py \
  --run "lightning_logs/LSTM_20260323_135531/version_0" \
  --run "lightning_logs/cLSTM_20260323_135847/version_0" \
  --run "lightning_logs/QLSTM_20260323_171749/version_0" \
  --run "lightning_logs/cQLSTM_20260323_135146/version_0" \
  --run "lightning_logs/SSM_20260323_140438/version_0" \
  --run "lightning_logs/QSSM_20260323_140529/version_0" \
  --run "lightning_logs/cQSSM_20260323_144546/version_0" \
  --run "lightning_logs/DeltaNet_20260323_151946/version_0" \
  --run "lightning_logs/QDeltaNet_20260323_165819/version_0" \
  2>&1 | grep -v "Warning\|pennylane\|Future\|Tip\|litlogger"


python scripts/eval.py \
  --run "lightning_logs/cQLSTM_20260323_174734/version_0" \
  --run "lightning_logs/cQLSTM_20260323_135146/version_0" \
  --run "lightning_logs/cQLSTM_20260323_174858/version_0" 2>&1 | grep -v "Warning\|pennylane\|Future\|Tip\|litlogger"



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

## Logging

```bash
# single run
tensorboard --logdir lightning_logs/cQLSTM_20260323_135146

# all runs
tensorboard --logdir lightning_logs --bind_all
```

Each run directory contains:
```
version_0/
├── best-epoch=XX-valloss=X.XXXX.ckpt   ← best checkpoint (monitored: val/loss)
├── run_config.yaml                      ← full resolved config (used by eval.py)
└── events.out.tfevents.*               ← TensorBoard scalars
```

Logged scalars per epoch: `train/loss`, `train/rmse`, `train/mae`, `train/r2`, `val/loss`, `val/rmse`, `val/mae`, `val/r2`, `test/*` (at end), per-layer gradient norms.


---

## License

MIT License — see [LICENSE](LICENSE).
