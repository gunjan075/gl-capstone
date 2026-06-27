# Insurance Price Prediction — Capstone

End-to-end machine-learning solution that estimates an individual's **`insurance_cost`** from
health, lifestyle, habit, and demographic variables. The project covers the full Great Learning
capstone scope — **Milestone 1** (data report, EDA, preprocessing) and **Milestone 2** (model
building, tuning, selection, deployment, business insights) — split into a clean two-track layout:

- **Experiment track** — Jupyter notebooks in [experiments/](experiments/) for EDA and modeling.
- **Production track** — a reusable Python package [src/insurance/](src/insurance/) that codifies the
  final pipeline, served through a **FastAPI** API, a **React** inference UI, and a thin **Streamlit** app.

## Final model

| Metric | Test value |
|---|---|
| **Model** | **LightGBM** (gradient boosting) |
| Test R² | **0.9585** |
| Test MAE | **2,328** |
| Test RMSE | **2,909** |
| Test MAPE | 11.3% |

Trained on an 80/20 split (`random_state=42`) with a leak-free scikit-learn `Pipeline`
(feature engineering → preprocessing → model). This beats the reference solution in
[anup/](anup/) (R² 0.958, MAE 2,335).

### Model comparison (top of [reports/tables/model_comparison.csv](reports/tables/model_comparison.csv))

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| **LightGBM** | **2,328** | **2,909** | **0.958** |
| HistGradientBoosting | 2,334 | 2,912 | 0.958 |
| CatBoost (GPU) | 2,345 | 2,929 | 0.958 |
| GradientBoosting | 2,368 | 2,944 | 0.957 |
| RandomForest | 2,381 | 3,006 | 0.956 |
| XGBoost (GPU) | 2,428 | 3,034 | 0.955 |
| LinearRegression | 2,710 | 3,356 | 0.945 |
| DummyRegressor (baseline) | 11,762 | 14,274 | 0.000 |

## Repository structure

```
gl-capstone/
├── data/raw/insurance.csv          # dataset (copy of templates/Insurance Data.csv)
├── src/insurance/                  # production pipeline package
│   ├── config.py                   # paths, column groups, GPU detection, risk thresholds
│   ├── data.py                     # load + schema cleanup (typo columns, "Salried")
│   ├── features.py                 # FeatureEngineer transformer (engineered features + imputation)
│   ├── preprocess.py               # ColumnTransformer (impute + scale + one-hot)
│   ├── pipeline.py                 # build_pipeline(model)
│   ├── models.py                   # model registry + hyperparameter search spaces (+ GPU config)
│   ├── evaluate.py                 # MAE/RMSE/R²/MAPE/SMAPE + cross-validation
│   ├── train.py                    # train → tune → select → persist  (python -m insurance.train)
│   ├── predict.py                  # load model, predict, risk category
│   └── schemas.py                  # pydantic I/O + form field metadata
├── experiments/                    # Milestone notebooks (01 EDA → 04 explainability)
├── api/main.py                     # FastAPI: /health, /schema, /predict
├── frontend/                       # React + Vite + TypeScript + Tailwind UI
├── streamlit_app.py                # thin Streamlit demo (capstone rubric)
├── models/                         # final_model.pkl, metrics.json
├── reports/{figures,tables}/       # EDA + diagnostic figures, comparison/importance tables
├── tests/                          # pytest (cleaning, features, predict)
└── pyproject.toml                  # uv-managed dependencies
```

## Setup

Python is managed with **[uv](https://docs.astral.sh/uv/)**; the frontend uses npm.

```bash
uv sync --extra dev          # creates .venv and installs all Python deps
cd frontend && npm install   # frontend deps
```

## Usage

All Python commands run through `uv run` (no manual venv activation).

```bash
# 1. Train the model (regenerates models/final_model.pkl + reports/)
uv run python -m insurance.train          # full run (use --quick for a fast smoke run)

# 2. Run the experiment notebooks
uv run jupyter lab experiments/

# 3. Serve the model
uv run uvicorn api.main:app --port 8000   # FastAPI  → http://localhost:8000/docs

# 4. React inference UI (in another terminal)
cd frontend && npm run dev                # → http://localhost:5173  (proxies /api to :8000)

# 5. Streamlit demo (capstone rubric)
uv run streamlit run streamlit_app.py
```

The React UI builds its form from the API's `/schema` endpoint, posts to `/predict`, and shows the
estimated cost, a **Low / Medium / High** risk badge, and the top model drivers.

## GPU acceleration

Training auto-detects an NVIDIA GPU (`config.USE_GPU`, overridable with `INSURANCE_USE_GPU=0/1`)
and trains **XGBoost** (`device="cuda"`) and **CatBoost** (`task_type="GPU"`) on it. LightGBM stays
on CPU (its pip wheel needs OpenCL, unavailable under WSL). A GPU-trained XGBoost is converted back
to CPU for the saved artifact so inference stays portable. To avoid GPU/thread contention, each
model is single-threaded and only the cross-validation / search loop is parallelised; GPU models
run sequentially.

## Milestone mapping

- **Milestone 1** — [01_milestone1_eda_preprocessing.ipynb](experiments/01_milestone1_eda_preprocessing.ipynb):
  data report, univariate/bivariate/multivariate EDA, missing-value & outlier treatment, feature
  engineering, scaling.
- **Milestone 2** — [02](experiments/02_milestone2_baseline_advanced_models.ipynb) baseline+advanced
  models, [03](experiments/03_milestone2_hyperparameter_tuning.ipynb) RandomizedSearch + Optuna +
  target-transform experiments, [04](experiments/04_milestone2_selection_explainability.ipynb)
  selection, explainability, business insights.

## Explainability & business view

- **Permutation importance** (production, in [feature_importance.csv](reports/tables/feature_importance.csv))
  and **SHAP** (notebook 04) agree: **`weight`** dominates, followed by admission history
  (`Year_last_admitted`), other-insurer coverage, and preventive checkups.
- **Decile gains / lift** (notebook 04, [decile_gains.csv](reports/tables/decile_gains.csv)) shows the
  model concentrates high-cost applicants in the top predicted deciles.

### Recommendations
1. Use predictions as **pricing decision-support**, not automated underwriting.
2. Route **extreme predictions** (top decile) to manual underwriting review.
3. Target **wellness programmes** (weight, checkups, activity) at high-risk segments.
4. **Governance**: review `Gender`/`Location` for proxy-discrimination, and validate the dominance
   of `weight` for plausibility/leakage before any production use.
5. **Monitor** drift, segment-level error, and override rates post-deployment.

## Limitations

- Dataset provenance and real pricing rules are unknown; `weight` dominates the signal and must be
  validated for plausibility/leakage before production.
- All predictors must be confirmed available at quote time.
- Fairness testing on sensitive fields should be expanded before real-world deployment.

## Testing

```bash
uv run pytest        # schema cleanup, feature engineering, prediction smoke test
```
