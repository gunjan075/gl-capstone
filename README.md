# Insurance Price Prediction Capstone

This project builds a supervised regression model to estimate `insurance_cost` using health, lifestyle, habit, and demographic variables.

## Reproducibility

Run the full pipeline from the project root:

```bash
python run_all.py
```

For grading, install the requirements and run `python run_all.py` from the project root.

The script generates EDA figures, summary tables, model artifacts, milestone reports, the final presentation, and the Streamlit app dependencies.

The target is a discrete business quote grid. The pipeline reports raw regression metrics, calibrated predictions, nearest valid quote-band outputs, and band-accuracy metrics.

## Notebooks and HTML exports

The submitted notebooks are executable end to end from `Insurance Data.csv`:

- Run `notebooks/01_milestone1_eda.ipynb` to recreate Milestone 1 EDA tables, figures, preprocessing evidence, and interpretations from the raw dataset.
- Run `notebooks/02_milestone2_modeling.ipynb` to recreate the train/test split, model training/comparison, calibration, quote-band evaluation, explainability artifacts, saved model, and Streamlit app from the raw dataset.

Milestone 2 runs the full model suite and may take several minutes on CPU. The notebooks do not require pre-generated output files; they create the required `outputs/` artifacts during execution.

`python run_all.py` can still be used to regenerate the full package plus executed HTML exports:

- `notebooks/01_milestone1_eda.ipynb`
- `notebooks/01_milestone1_eda.html`
- `notebooks/02_milestone2_modeling.ipynb`
- `notebooks/02_milestone2_modeling.html`
- `outputs/notebooks/01_milestone1_eda.html`
- `outputs/notebooks/02_milestone2_modeling.html`

The Milestone 1 notebook follows the rubric sections for Data Report, Initial EDA, Data Pre-processing, and Extensive EDA. The HTML copies are included for reviewers who want to inspect executed notebook output without opening Jupyter.

## Final model

Selected model: `BaseHistGradientBoostingRegressorAlt`

- Test MAE: 2,394
- Test RMSE: 2,982
- Test R2: 0.956

## Streamlit

```bash
streamlit run app.py
```

The app loads `outputs/models/final_model.pkl` and applies the same preprocessing pipeline used during training.
It also reads `outputs/models/app_schema.json` and, when available, `outputs/models/prediction_calibrator.pkl` to show raw predicted cost, calibrated cost, and nearest valid quote band.
Raw prediction is used for the analytical estimate and default risk category; the nearest quote band is rounded from the selected deployment variant in `outputs/models/model_metadata.json`. Calibrated cost is shown as a secondary diagnostic unless metadata explicitly chooses it for deployment.

## Key generated validation artifacts

- `outputs/models/repeated_cv_metrics.csv`
- `outputs/models/final_model_summary.csv`
- `outputs/models/price_grid_evaluation.csv`
- `outputs/models/calibration_comparison.csv`
- `outputs/models/ordinal_challenger_metrics.csv`
- `outputs/models/app_schema.json`

## Colab GPU Training

Use `notebooks/03_colab_gpu_training.ipynb` in Google Colab with a GPU runtime to run the heavier optional XGBoost/CatBoost/LightGBM comparison without using this PC. The notebook expects the prepared project zip from `outputs/colab/insurance_capstone_colab_bundle.zip`, runs `python run_all.py` with `INSURANCE_USE_GPU=1`, and downloads the regenerated artifacts.

Create or refresh the upload bundle with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_colab_bundle.ps1
```
