# Hydraulic System Condition Analysis

Classifies the condition of three components of a hydraulic test rig — **cooler**, **valve** and **pump** — from one 60-second load cycle of 17 sensor signals, and serves the predictions through a REST API.

| Stage | What it does | Code |
|---|---|---|
| Data | Loads 17 sensor files (pressure, flow, temperature, vibration, power, cooling) and the condition labels; checks alignment across cycles and rejects NaN/inf | `src/hydraulic/data.py` |
| Features | Vectorised NumPy statistics per cycle: mean, std, min, max → 17 × 4 = 68 features | `src/hydraulic/features.py` |
| Models | Logistic regression, k-NN and random forest per component; compared on a stratified train/test split; best one refit and saved | `src/hydraulic/model.py` |
| Reports | Accuracy comparison bar chart and confusion matrices (Matplotlib) | `src/hydraulic/train.py` |
| Serving | FastAPI with Pydantic request models; 422 on invalid input, 503 if no model; prediction logic decoupled from the API | `src/hydraulic/api.py` |
| Tests | 24 pytest tests: data validation, feature maths, model selection, API error paths | `tests/` |

## Dataset

UCI Machine Learning Repository: *Condition monitoring of hydraulic systems* (Helwig, Pignanelli, Schütze, 2015) — 2,205 load cycles.

| Sensor | Quantity | Rate | Samples/cycle |
|---|---|---|---|
| PS1–PS6 | Pressure [bar] | 100 Hz | 6000 |
| EPS1 | Motor power [W] | 100 Hz | 6000 |
| FS1–FS2 | Volume flow [l/min] | 10 Hz | 600 |
| TS1–TS4 | Temperature [°C] | 1 Hz | 60 |
| VS1 | Vibration [mm/s] | 1 Hz | 60 |
| CE, CP, SE | Cooling efficiency, cooling power, efficiency factor | 1 Hz | 60 |

Targets (from `profile.txt`):

- **cooler**: 3 = close to total failure, 20 = reduced efficiency, 100 = full efficiency
- **valve**: 100 = optimal, 90 = small lag, 80 = severe lag, 73 = close to total failure
- **pump**: 0 = no leakage, 1 = weak leakage, 2 = severe leakage

Download the dataset from the UCI repository and unzip the `.txt` files into `data/raw/`.

A **synthetic generator** (`make_synthetic`) produces data in the same layout with plausible fault physics (weak cooler → higher temperature, valve lag → slower pressure build-up, pump leakage → lower flow and higher motor power). It is used by the tests and for running the pipeline without the download. It is deliberately easy to separate, so accuracy on it says nothing about the real data.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# train
hydraulic-train --data data/raw        # real dataset
hydraulic-train --synthetic            # no download needed

# test
pytest

# serve
uvicorn hydraulic.api:app --reload
# docs at http://127.0.0.1:8000/docs
```

## API

| Method | Path | Body |
|---|---|---|
| GET | `/health` | – |
| POST | `/predict/cycle` | `{"signals": {"PS1": [...], ..., "SE": [...]}}` — raw samples of one cycle, all 17 sensors |
| POST | `/predict/features` | `{"features": {"PS1_mean": 160.1, ...}}` — the 68 precomputed features |

Example response:

```json
{
  "cooler": {"label": 100, "description": "full efficiency", "confidence": 0.998},
  "valve":  {"label": 80,  "description": "severe lag",      "confidence": 0.977},
  "pump":   {"label": 2,   "description": "severe leakage",  "confidence": 0.993}
}
```

Error handling: missing/unknown sensor, fewer than 2 samples, non-numeric or non-finite values, or missing features → **422** with the offending name in the message. No trained model on disk → **503**.

## Docker

```bash
hydraulic-train --synthetic            # or --data data/raw
docker build -t hydraulic-api .
docker run -p 8000:8000 hydraulic-api
```

## Project layout

```
src/hydraulic/
  config.py     sensor rates, label meanings
  data.py       loading, validation, synthetic generator
  features.py   per-cycle statistics
  model.py      training, comparison, ConditionPredictor
  train.py      CLI + plots
  api.py        FastAPI service
tests/          pytest suite
models/         saved model bundle (generated)
reports/        plots (generated)
```

## Possible extensions

- Frequency-domain features (FFT band energy on PS/EPS1) and slope features for valve lag
- Cross-validation and hyperparameter search instead of a single split
- Accumulator and stable-flag targets
- Feature importance plot to show which sensors drive each prediction
