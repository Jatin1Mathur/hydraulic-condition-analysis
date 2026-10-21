# Project Guide

A complete explanation of this project: what it does, how it was built step by step, the concepts behind it, the tools used and how, and how to talk about it in an interview. For the technical structure and diagrams, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. The problem

A hydraulic machine moves force with pressurised oil: a **pump** pushes the oil, a **valve** controls it, a **cooler** keeps it from overheating, and an **accumulator** stores pressure and smooths out pulses. When any of these degrade, the machine becomes less efficient and eventually fails.

**Condition monitoring** means watching sensor data to judge the health of each part. It is the basis of **predictive maintenance**: repairing a part shortly before it fails, instead of on a fixed schedule (wasteful) or after it breaks (expensive downtime).

This project answers one question: *given the sensor data from one 60-second working cycle, what condition is each of the four components in?*

## 2. The dataset

The UCI dataset *Condition monitoring of hydraulic systems* (Helwig, Pignanelli, Schütze, ZeMA, 2015) comes from a real test rig. The researchers deliberately put components into known states of damage and recorded 2,205 cycles.

| Sensors | What they measure | Samples per cycle |
|---|---|---|
| PS1–PS6 | Pressure | 6,000 (100 Hz) |
| EPS1 | Motor power | 6,000 (100 Hz) |
| FS1–FS2 | Oil flow | 600 (10 Hz) |
| TS1–TS4 | Temperature | 60 (1 Hz) |
| VS1 | Vibration | 60 (1 Hz) |
| CE, CP, SE | Cooling efficiency, cooling power, system efficiency | 60 (1 Hz) |

| Component | States (labels) |
|---|---|
| Cooler | 3 close to failure · 20 reduced · 100 full efficiency |
| Valve | 73 close to failure · 80 severe lag · 90 small lag · 100 optimal |
| Pump | 0 no leakage · 1 weak · 2 severe leakage |
| Accumulator | 90 close to failure · 100 severely reduced · 115 slightly reduced · 130 optimal |

Each fault leaves a physical fingerprint: a weak cooler makes the oil hotter, a lagging valve slows pressure build-up, a leaking pump reduces flow while the motor works harder, and a weak accumulator damps pressure pulses less.

## 3. What was built, step by step

### Step 1: Project from the CV description

The project started from the CV entry *Hydraulic System Condition Analysis* and turned each bullet into working code:

1. A Python package with a `src/` layout: `config`, `data`, `features`, `model`, `train`, `api`.
2. A data loader for the 17 UCI files with validation (alignment, NaN, missing sensors).
3. A **synthetic data generator** that writes files in the exact UCI format, with simple fault physics, so the pipeline and tests run without the real data.
4. Feature extraction: mean, std, min and max per sensor (68 features).
5. Three classifiers per component (logistic regression, k-NN, random forest) compared on a train/test split.
6. A FastAPI service with Pydantic validation and clear error codes.
7. 24 pytest tests, a README, `pyproject.toml`, a Dockerfile and `.gitignore`.
8. Pushed to GitHub as the first commit.

### Step 2: Getting the real data

UCI and Kaggle downloads were blocked from the build environment, so the identical dataset was taken from a public GitHub copy. It was verified before use: 2,205 rows in every file, the correct number of samples per sensor, and label counts matching the official description exactly (e.g. cooler 732/732/741, valve 360/360/360/1125).

The loader was switched from `np.loadtxt` to pandas' C parser, which is about 20 times faster on the 90 MB pressure files.

### Step 3: Baseline on real data

The original pipeline scored cooler 100 %, valve 98.2 %, pump 99.8 % on a random 20 % test split. These looked excellent, but two weaknesses were suspected:

1. The best model was chosen by its score on the same test set that was reported, which biases the number upward.
2. Neighbouring cycles are nearly identical, so a random split puts near-copies of test cycles into training.

### Step 4: The experiment that changed the project

The label order was examined: a valve state lasts about 15 consecutive cycles, a pump state about 60, while the cooler has only 3 states in total. Two evaluation methods were compared:

- **Random (stratified) cross-validation:** shuffled folds.
- **Time-blocked cross-validation:** each fold is a continuous period of cycles.

| Features | Valve, random CV | Valve, time-blocked CV |
|---|---|---|
| 68 whole-cycle statistics | ~98 % | **72.9 % ± 12.9** |
| 204 with time-window, slope, roughness | ~99.9 % | **98.4 % ± 1.6** |

The random split was hiding a 25-point gap. With only averages and extremes, the model was partly recognising neighbouring cycles rather than learning valve behaviour. Valve lag changes *when* pressure rises during the cycle, not its average, so features that describe the signal over time were added. That fixed it.

### Step 5: Rebuilding the pipeline around the finding

- **Features: 68 → 204.** 12 per sensor: mean, std, min, max, six 10-second window means, slope and mean absolute change.
- **Evaluation protocol:** hold out 20 % as a test set; choose the model by 5-fold CV on the other 80 % (time-blocked for valve and pump, stratified for cooler and accumulator); score the chosen model once on the test set; then refit on all data.
- **Models:** logistic regression, random forest, extra trees and gradient boosting (k-NN was dropped as consistently weakest).
- **Accumulator** added as a fourth component in the model, API, synthetic data and tests.
- **Reports:** model comparison chart, confusion matrices, a sensor importance chart and `results.json`.
- **Tests:** 24 → 31.
- README rewritten with real results; pushed as the second commit.

### Step 6: Documentation

This guide and [ARCHITECTURE.md](ARCHITECTURE.md) were added.

## 4. Final results

| Component | Selected model | CV accuracy | Test accuracy | Macro F1 |
|---|---|---|---|---|
| Cooler | Gradient boosting | 99.89 % ± 0.14 | 100.0 % | 1.000 |
| Valve | Gradient boosting | 98.35 % ± 1.63 (time-blocked) | 99.1 % | 0.988 |
| Pump | Logistic regression | 97.90 % ± 2.51 (time-blocked) | 100.0 % | 1.000 |
| Accumulator | Extra trees | 99.21 % ± 0.42 | 99.1 % | 0.990 |

Most important sensors per component: cooler → cooling efficiency (CE); valve → pressure (PS2) and flow (FS1); pump → efficiency (SE), flow (FS1), motor power (EPS1); accumulator → spread across many sensors.

## 5. Concepts

### Data
- **Sampling rate:** how often a sensor is read. Different rates give arrays of different lengths that cannot be stacked directly.
- **Data validation:** checking shape, alignment and missing values before using data, and failing early with a clear message.
- **Synthetic data:** generated data with the same structure as the real data, used for fast and reproducible tests.

### Feature engineering
- **Feature:** a number that describes an example, here one cycle.
- **Summary statistics** (mean, std, min, max): level, spread and extremes of a signal.
- **Window features:** the cycle cut into six parts with the mean of each part; they describe the shape of the signal over time.
- **Slope:** the trend of a straight line fitted through the signal.
- **Roughness (mean absolute difference):** how much the signal jumps between samples.
- **Vectorisation:** applying an operation to a whole array at once with NumPy instead of a Python loop; much faster.

### Machine learning
- **Supervised classification:** learning from labelled examples to predict a category.
- **Multi-class:** more than two possible labels (3 or 4 here).
- **Logistic regression:** a linear model that outputs class probabilities; needs scaled features.
- **Decision tree:** a sequence of yes/no questions about features.
- **Random forest:** many trees trained on random samples of data and features; they vote.
- **Extra trees:** like a random forest but with random split thresholds; lower variance.
- **Gradient boosting:** trees built one after another, each correcting the previous ones' mistakes.
- **Feature scaling (StandardScaler):** shifting and scaling features to mean 0 and standard deviation 1, so large-valued features do not dominate.
- **Pipeline:** chaining preprocessing and model so they are fitted and applied together, preventing leakage of test information into scaling.

### Evaluation
- **Train/test split:** keeping data the model never sees, to measure generalisation.
- **Overfitting:** a model that memorises training data and performs worse on new data.
- **Stratification:** keeping class proportions equal in every split.
- **Cross-validation (k-fold):** splitting data into k parts, training on k−1 and testing on the remaining one, k times; gives a mean and spread instead of a single lucky or unlucky number.
- **Time-blocked cross-validation:** folds made of consecutive cycles, so the model is tested on unseen periods.
- **Data leakage:** information from the test set reaching training. Here it appeared as near-duplicate neighbouring cycles, and as choosing the model with the test set.
- **Model selection bias:** reporting the score of the test set used to choose the model overstates performance.
- **Accuracy:** share of correct predictions.
- **Macro F1:** the average of per-class F1 scores, so small classes count as much as large ones.
- **Confusion matrix:** a table of actual vs predicted classes that shows which states get confused.
- **Feature importance:** how much each feature (here grouped by sensor) contributes to a tree model's decisions.

### Software engineering
- **Separation of concerns:** each module does one job.
- **REST API:** programs exchange JSON over HTTP using methods like GET and POST.
- **Request validation:** rejecting bad input at the boundary with a clear message.
- **HTTP status codes:** 200 success, 422 invalid input, 503 service unavailable.
- **Dependency injection:** a function receives what it needs instead of creating it, so tests can swap in substitutes.
- **Caching:** keeping the loaded model in memory instead of reading it on every request.
- **Training/serving skew:** when training and serving compute features differently; avoided by sharing one function.
- **Unit and negative testing:** checking correct behaviour and that bad input fails safely (fault injection).
- **Reproducibility:** fixed random seeds give identical results each run.
- **Containerisation:** packaging code and model so they run the same everywhere.

## 6. Tools and how they were used

| Tool | Used for | Where |
|---|---|---|
| Python 3 | Whole project | everywhere |
| NumPy | Vectorised features along `axis=1`; slope via matrix product; synthetic signals | `features.py`, `data.py` |
| pandas | Fast parsing of the large text files; feature table; label analysis | `data.py`, `features.py` |
| scikit-learn | `LogisticRegression`, `RandomForestClassifier`, `ExtraTreesClassifier`, `HistGradientBoostingClassifier`, `StandardScaler`, `make_pipeline`, `train_test_split`, `KFold`, `StratifiedKFold`, `cross_val_score`, `accuracy_score`, `f1_score`, `confusion_matrix` | `model.py` |
| joblib | Saving and loading the trained model bundle | `model.py` |
| Matplotlib | Model comparison, confusion matrices, sensor importance plots (headless `Agg` backend) | `train.py` |
| FastAPI | REST endpoints, dependency injection, automatic Swagger docs | `api.py` |
| Pydantic | Request and response models, custom validator for sensor names and sample counts | `api.py` |
| Uvicorn | ASGI server that runs the FastAPI app | Dockerfile |
| pytest | 31 tests, session-scoped fixtures, `tmp_path`, `monkeypatch` | `tests/` |
| FastAPI TestClient (httpx) | Calling the API in tests without a running server | `test_api.py` |
| setuptools / `pyproject.toml` | Package definition, dependencies, `hydraulic-train` command | root |
| Docker | Container image for the API | `Dockerfile` |
| Git / GitHub | Version control and hosting, with descriptive commit messages | repository |

## 7. Repository contents

```
hydraulic-condition-analysis/
├── README.md                 results, quick start, API
├── docs/
│   ├── ARCHITECTURE.md       diagrams and design decisions
│   └── PROJECT_GUIDE.md      this file
├── src/hydraulic/
│   ├── config.py             sensors, labels, CV scheme per component
│   ├── data.py               loading, validation, synthetic data
│   ├── features.py           204 features per cycle
│   ├── model.py              model selection, evaluation, predictor
│   ├── train.py              command-line training, plots, results.json
│   └── api.py                FastAPI service
├── tests/                    31 pytest tests
├── reports/                  plots and results.json from the real-data run
├── data/                     put the raw UCI files in data/raw/ (not committed)
├── models/                   trained model bundle (generated, not committed)
├── pyproject.toml
├── Dockerfile
└── .gitignore
```

## 8. Interview preparation

**Describe the project in 30 seconds.**
I built a system that detects faults in four parts of a hydraulic machine from 17 sensor signals. It reaches 99–100 % test accuracy and serves predictions through a tested FastAPI service. The most interesting part was discovering that a random test split hid a real weakness: tested on unseen time periods, the valve model was only 73 % accurate. Adding time-based features raised it to 98 %.

**Why not feed the raw signals to the model?**
6,000 values per pressure sensor, at different rates per sensor, give about 44,000 inputs per cycle for only 2,205 examples. Summarising each sensor into 12 features solves the different sampling rates, keeps the model small, and makes the features interpretable.

**Why did the random split overestimate accuracy?**
Consecutive cycles are almost identical. With random splitting, a near-copy of each test cycle is in the training set, so the model can succeed by recognising neighbours. Time-blocked folds remove that shortcut.

**Why not use time-blocked validation for every component?**
The cooler has only three states, each in one continuous block. Any held-out time block would contain a state the model never saw during training, so the test would be meaningless.

**Why did window features help the valve specifically?**
A lagging valve changes how quickly pressure builds up within the cycle. The mean and extremes stay similar, but the early windows differ. A test in the suite shows this directly: two signals with identical statistics in a different order get different window features.

**Why choose the model by cross-validation and not the test set?**
If the test set influences the choice, its score is no longer an unbiased estimate. Choosing by CV on training data and scoring the test set once keeps the final number honest.

**Why is the prediction logic separate from the API?**
So it can be tested without HTTP, reused from scripts or batch jobs, and kept if the web framework changes. The API only validates input and translates errors to status codes.

**How do you know the model is not cheating?**
The sensor importances match the physics: cooling efficiency for the cooler, pressure and flow for the valve, flow and motor power for the pump. The time-blocked results confirm the model generalises across periods.

**What would you do next?**
Frequency features (FFT) of pressure and motor power, hyperparameter search inside cross-validation, and testing on data from a different machine, which is the real test for deployment.

**What are the limitations?**
All data comes from one test rig with deliberately induced faults. Real machines have gradual wear, other operating conditions and sensor drift, so the model would need validation and probably retraining on field data.
