from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Insurance_Price_Prediction_Project_Report.html"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "outputs" / "models"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)


def fmt_num(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if abs(number) >= 1000:
        return f"{number:,.{digits}f}"
    return f"{number:.{digits}f}"


def df_to_html_table(df: pd.DataFrame, max_rows: int | None = None, digits: int = 2) -> str:
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in view.columns)
    rows = []
    for _, row in view.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f"<td class='num'>{fmt_num(value, digits)}</td>")
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    if max_rows is not None and len(df) > max_rows:
        rows.append(
            f"<tr><td colspan='{len(view.columns)}' class='muted'>Showing {max_rows} of {len(df)} rows. Full table is saved under outputs/tables.</td></tr>"
        )
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def image_data_uri(name: str) -> str:
    path = FIG_DIR / name
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def figure(name: str, caption: str) -> str:
    return f"""
    <figure>
      <img src="{image_data_uri(name)}" alt="{html.escape(caption)}" loading="lazy" />
      <figcaption>{html.escape(caption)}</figcaption>
    </figure>
    """


def stat_card(label: str, value: str, detail: str = "") -> str:
    return f"""
      <div class="stat-card">
        <span>{html.escape(label)}</span>
        <strong>{html.escape(value)}</strong>
        <small>{html.escape(detail)}</small>
      </div>
    """


def artifact_link(path: str, label: str) -> str:
    return f"<a href='{html.escape(path)}'>{html.escape(label)}</a>"


def main() -> None:
    raw = pd.read_csv(ROOT / "Insurance Data.csv")
    profile = read_csv("dataset_profile.csv")
    missing = read_csv("missing_values.csv")
    data_dictionary = read_csv("data_dictionary_summary.csv")
    categorical = read_csv("categorical_frequency.csv")
    rename_cleanup = read_csv("renaming_and_cleanup.csv")
    outliers = read_csv("outlier_summary.csv")
    metrics = pd.read_csv(MODEL_DIR / "model_metrics.csv")
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv")
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    previous_run_comparison = (
        pd.read_csv(MODEL_DIR / "previous_run_comparison.csv")
        if (MODEL_DIR / "previous_run_comparison.csv").exists()
        else pd.DataFrame()
    )
    price_grid_evaluation = (
        pd.read_csv(MODEL_DIR / "price_grid_evaluation.csv")
        if (MODEL_DIR / "price_grid_evaluation.csv").exists()
        else pd.DataFrame()
    )
    calibration_comparison = (
        pd.read_csv(MODEL_DIR / "calibration_comparison.csv")
        if (MODEL_DIR / "calibration_comparison.csv").exists()
        else pd.DataFrame()
    )
    final_model_summary = (
        pd.read_csv(MODEL_DIR / "final_model_summary.csv")
        if (MODEL_DIR / "final_model_summary.csv").exists()
        else pd.DataFrame()
    )
    residual_segments = (
        pd.read_csv(TABLE_DIR / "residual_segment_summary.csv")
        if (TABLE_DIR / "residual_segment_summary.csv").exists()
        else pd.DataFrame()
    )
    fairness_summary = (
        pd.read_csv(TABLE_DIR / "fairness_error_summary.csv")
        if (TABLE_DIR / "fairness_error_summary.csv").exists()
        else pd.DataFrame()
    )
    missing_admission_profile = (
        pd.read_csv(TABLE_DIR / "missing_year_last_admitted_profile.csv")
        if (TABLE_DIR / "missing_year_last_admitted_profile.csv").exists()
        else pd.DataFrame()
    )
    missing_bmi_profile = (
        pd.read_csv(TABLE_DIR / "missing_bmi_profile.csv")
        if (TABLE_DIR / "missing_bmi_profile.csv").exists()
        else pd.DataFrame()
    )
    target_tail_profile = (
        pd.read_csv(TABLE_DIR / "target_tail_profile.csv")
        if (TABLE_DIR / "target_tail_profile.csv").exists()
        else pd.DataFrame()
    )

    target = raw["insurance_cost"]
    selected_metrics = metrics.loc[metrics["model"] == metadata["final_model"]]
    best = selected_metrics.iloc[0] if not selected_metrics.empty else metrics.iloc[0]
    best_test_metrics = metrics.sort_values(["test_RMSE", "test_MAE"]).iloc[0]
    missing_nonzero = missing[missing["missing_count"] > 0].copy()
    top_importance = importance.head(8)
    top_feature_text = ", ".join(top_importance["feature"].astype(str).tolist())
    cat_cols = [
        "Occupation",
        "cholesterol_level",
        "Gender",
        "smoking_status",
        "Location",
        "covered_by_any_other_company",
        "Alcohol",
        "exercise",
    ]

    group_tables = []
    for name in [
        "smoking_status",
        "exercise",
        "Alcohol",
        "covered_by_any_other_company",
        "adventure_sports",
        "any_major_disease_history",
        "bmi_category",
        "age_band",
        "lifestyle_risk_band",
        "medical_risk_band",
        "risk_profile_segment",
    ]:
        path = TABLE_DIR / f"group_summary_{name}.csv"
        if path.exists():
            group_tables.append((name, pd.read_csv(path)))

    best_params = metadata.get("best_params", {})
    final_params = (
        pd.DataFrame([{"parameter": k, "value": v} for k, v in best_params.items()])
        if best_params
        else pd.DataFrame([{"parameter": "configuration", "value": "Selected from model comparison without extra tuning"}])
    )
    target_grid = metadata.get("target_grid", {})
    deployment_variant = metadata.get("deployment_variant", {})
    optional_packages = metadata.get("external_model_packages", {})
    available_boosters = ", ".join(
        name for name, available in optional_packages.items() if available and name in {"xgboost", "lightgbm", "catboost"}
    ) or "sklearn models only"
    shap_status = metadata.get("shap_status", {}).get("status", "not_available")

    css = """
    :root {
      --ink: #14213d;
      --muted: #5f6b7a;
      --line: #d9e2ec;
      --paper: #ffffff;
      --bg: #f6f8fb;
      --accent: #2a9d8f;
      --accent-2: #e76f51;
      --accent-soft: #e8f5f2;
      --warn-soft: #fff1ed;
      --shadow: 0 18px 45px rgba(20, 33, 61, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: "Aptos", "Segoe UI", Arial, sans-serif;
      line-height: 1.55;
    }
    a { color: #0f766e; text-decoration: none; font-weight: 700; }
    a:hover { text-decoration: underline; }
    .hero {
      background:
        linear-gradient(135deg, rgba(20,33,61,.92), rgba(20,33,61,.78)),
        radial-gradient(circle at 80% 15%, rgba(42,157,143,.35), transparent 28%),
        linear-gradient(135deg, #14213d, #2a9d8f);
      color: #fff;
      padding: 56px 6vw 48px;
    }
    .hero-grid {
      max-width: 1180px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1.4fr .9fr;
      gap: 32px;
      align-items: end;
    }
    .eyebrow {
      letter-spacing: .12em;
      text-transform: uppercase;
      font-size: 12px;
      color: #a7f3d0;
      font-weight: 800;
    }
    h1 {
      font-size: clamp(40px, 6vw, 76px);
      line-height: .95;
      margin: 12px 0 18px;
      max-width: 780px;
    }
    .hero p {
      color: #dbe7ef;
      font-size: 18px;
      max-width: 780px;
      margin: 0;
    }
    .hero-panel {
      background: rgba(255, 255, 255, .10);
      border: 1px solid rgba(255, 255, 255, .24);
      border-radius: 10px;
      padding: 22px;
      backdrop-filter: blur(8px);
    }
    .hero-panel strong { display: block; font-size: 34px; line-height: 1; margin-bottom: 8px; }
    .layout {
      max-width: 1180px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 260px 1fr;
      gap: 28px;
      padding: 34px 22px 64px;
    }
    nav {
      position: sticky;
      top: 16px;
      align-self: start;
      background: var(--paper);
      box-shadow: var(--shadow);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 18px;
    }
    nav h3 { margin: 0 0 12px; font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
    nav a { display: block; padding: 7px 0; color: var(--ink); font-size: 14px; font-weight: 650; }
    main { min-width: 0; }
    section {
      background: var(--paper);
      box-shadow: var(--shadow);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 30px;
      margin-bottom: 24px;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 30px;
      line-height: 1.12;
    }
    h3 {
      margin: 26px 0 10px;
      font-size: 20px;
    }
    .section-lead {
      color: var(--muted);
      font-size: 16px;
      margin: 0 0 22px;
    }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 20px 0;
    }
    .stat-card {
      background: #f8fbfd;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 16px;
      min-height: 110px;
    }
    .stat-card span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 800;
      letter-spacing: .06em;
    }
    .stat-card strong {
      display: block;
      font-size: 28px;
      margin: 8px 0 6px;
      color: var(--ink);
    }
    .stat-card small { color: var(--muted); }
    .callout {
      border-left: 5px solid var(--accent);
      background: var(--accent-soft);
      padding: 16px 18px;
      border-radius: 10px;
      margin: 18px 0;
    }
    .warning {
      border-left-color: var(--accent-2);
      background: var(--warn-soft);
    }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }
    .grid-3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
    }
    figure {
      margin: 18px 0;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: #fff;
    }
    figure img {
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
    }
    figcaption {
      padding: 11px 14px;
      color: var(--muted);
      font-size: 13px;
      border-top: 1px solid var(--line);
      background: #fbfdff;
    }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      margin: 14px 0 20px;
      background: white;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      min-width: 680px;
    }
    th {
      background: var(--ink);
      color: white;
      text-align: left;
      padding: 10px 12px;
      position: sticky;
      top: 0;
    }
    td {
      border-top: 1px solid var(--line);
      padding: 9px 12px;
      vertical-align: top;
    }
    tr:nth-child(even) td { background: #fbfdff; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    ul, ol { padding-left: 22px; }
    li { margin: 7px 0; }
    .tag {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      margin: 4px 4px 4px 0;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }
    .deliverables {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .deliverables a {
      display: block;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fbfdff;
    }
    code {
      background: #edf2f7;
      padding: 2px 5px;
      border-radius: 5px;
      color: #0f172a;
    }
    pre {
      background: #0f172a;
      color: #e5e7eb;
      padding: 16px;
      border-radius: 10px;
      overflow-x: auto;
    }
    .muted { color: var(--muted); }
    .footer {
      text-align: center;
      color: var(--muted);
      padding: 20px 0 4px;
      font-size: 13px;
    }
    @media (max-width: 920px) {
      .hero-grid, .layout, .grid-2, .grid-3, .deliverables { grid-template-columns: 1fr; }
      nav { position: static; }
      .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media print {
      nav { display: none; }
      .layout { display: block; max-width: none; padding: 0; }
      section { box-shadow: none; page-break-inside: avoid; }
      .hero { print-color-adjust: exact; }
    }
    """

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Insurance Price Prediction Capstone - Project Report</title>
  <style>{css}</style>
</head>
<body>
  <header class="hero">
    <div class="hero-grid">
      <div>
        <div class="eyebrow">Capstone Project Report</div>
        <h1>Insurance Price Prediction</h1>
        <p>End-to-end machine learning solution for estimating <code>insurance_cost</code> from health, lifestyle, habit, and demographic variables. This report consolidates the problem statement, dataset audit, EDA, preprocessing, model building, final model selection, deployment workflow, limitations, and business recommendations.</p>
      </div>
      <div class="hero-panel">
        <strong>{metadata["final_model"]}</strong>
        <div>Final selected model</div>
        <hr style="border:0;border-top:1px solid rgba(255,255,255,.25);margin:18px 0" />
        <div>Test MAE: <b>{fmt_num(best["test_MAE"], 0)}</b></div>
        <div>Test RMSE: <b>{fmt_num(best["test_RMSE"], 0)}</b></div>
        <div>Test R2: <b>{float(best["test_R2"]):.3f}</b></div>
      </div>
    </div>
  </header>

  <div class="layout">
    <nav>
      <h3>Contents</h3>
      <a href="#summary">Executive Summary</a>
      <a href="#problem">Problem and Objective</a>
      <a href="#data">Dataset Audit</a>
      <a href="#preprocessing">Preprocessing</a>
      <a href="#eda">EDA Insights</a>
      <a href="#modeling">Modeling</a>
      <a href="#explainability">Explainability</a>
      <a href="#deployment">Deployment</a>
      <a href="#recommendations">Business Recommendations</a>
      <a href="#risks">Risks and Limitations</a>
      <a href="#files">Files and Reproducibility</a>
    </nav>

    <main>
      <section id="summary">
        <h2>Executive Summary</h2>
        <p class="section-lead">The project implements a complete supervised regression workflow for estimating individual insurance cost and translating model outputs into business-ready recommendations.</p>
        <div class="stat-grid">
          {stat_card("Rows", f"{len(raw):,}", "No duplicate rows detected")}
          {stat_card("Columns", f"{raw.shape[1]:,}", "24 original variables")}
          {stat_card("Target mean", fmt_num(target.mean(), 0), "insurance_cost")}
          {stat_card("Target skew", f"{target.skew():.3f}", "Mild right skew")}
          {stat_card("Final MAE", fmt_num(best["test_MAE"], 0), "Average absolute pricing error")}
          {stat_card("Final RMSE", fmt_num(best["test_RMSE"], 0), "Penalizes larger pricing errors")}
          {stat_card("Final R2", f"{float(best['test_R2']):.3f}", "Variance explained on test set")}
          {stat_card("Quote bands", fmt_num(target_grid.get("target_unique_count", ""), 0), f"Step {fmt_num(target_grid.get('target_grid_step', ''), 0)}")}
          {stat_card("Streamlit", "Ready", "Uses saved sklearn pipeline")}
        </div>
        <div class="callout">
          <b>Bottom line:</b> The selected final model is <b>{metadata["final_model"]}</b>. It explains about <b>{float(best["test_R2"]) * 100:.1f}%</b> of test-set variance and produces an average absolute error of about <b>{fmt_num(best["test_MAE"], 0)}</b> cost units. The best raw test-RMSE model is reported separately as <b>{best_test_metrics["model"]}</b>.
        </div>
      </section>

      <section id="problem">
        <h2>Business Problem and Objective</h2>
        <p>The insurance company needs a data-driven way to estimate an appropriate insurance cost for each applicant. The target variable is <code>insurance_cost</code>, and predictors include profile variables, lifestyle habits, health measures, disease-history fields, prior admission information, and behavior indicators.</p>
        <h3>Primary Objective</h3>
        <p>Build, evaluate, explain, and deploy a regression model that estimates <code>insurance_cost</code> accurately enough to support pricing decisions, manual review triage, and wellness-oriented business interventions.</p>
        <h3>Sub-objectives</h3>
        <ul>
          <li>Load and audit the supplied CSV dataset.</li>
          <li>Correct schema and category quality issues without losing business meaning.</li>
          <li>Treat missing values and outliers using defensible preprocessing logic.</li>
          <li>Engineer features that represent admission recency, BMI risk, cholesterol level, disease history, and activity intensity.</li>
          <li>Compare baseline, linear, regularized, tree, and ensemble regression models.</li>
          <li>Select a final model using MAE, RMSE, R2, MAPE, cross-validation stability, train-test gap, and a parsimony rule.</li>
          <li>Deploy a Streamlit interface that loads the saved model and returns raw cost, nearest quote band, optional calibrated cost, and risk category.</li>
        </ul>
      </section>

      <section id="data">
        <h2>Dataset Audit</h2>
        <p class="section-lead">The source file is <code>Insurance Data.csv</code>. The dataset has 25,000 rows and 24 original columns.</p>
        <div class="callout">
          <b>Target-grid finding:</b> <code>insurance_cost</code> contains {target_grid.get("target_unique_count")} valid quote bands from {fmt_num(target_grid.get("target_min"), 0)} to {fmt_num(target_grid.get("target_max"), 0)}, with a grid step of {fmt_num(target_grid.get("target_grid_step"), 0)}.
        </div>
        <h3>Dataset Profile</h3>
        {df_to_html_table(profile, digits=3)}
        <h3>Missing Values</h3>
        <p>Only two fields contain missing values. <code>Year_last_admitted</code> has high missingness and is treated as meaningful admission-history information. <code>bmi</code> has moderate missingness and is imputed.</p>
        {df_to_html_table(missing_nonzero, digits=3)}
        {figure("missing_values.png", "Missing values by column. Year_last_admitted and BMI are the only incomplete fields.")}
        <h3>Missingness Profiles</h3>
        <p>Missingness was profiled to check whether incomplete admission or BMI fields describe materially different customer groups.</p>
        {df_to_html_table(missing_admission_profile, digits=2) if not missing_admission_profile.empty else ""}
        {df_to_html_table(missing_bmi_profile, digits=2) if not missing_bmi_profile.empty else ""}
        <h3>Target Tail Profile</h3>
        <p>The top and bottom 1% of insurance costs were profiled instead of removed, because extreme premiums can be valid business cases.</p>
        {df_to_html_table(target_tail_profile, digits=2) if not target_tail_profile.empty else ""}
        <h3>Data Dictionary and Field Audit</h3>
        {df_to_html_table(data_dictionary, max_rows=30, digits=3)}
        <h3>Categorical Frequency Summary</h3>
        <p>The dataset includes categorical variables for occupation, cholesterol range, gender, smoking status, location, other insurance coverage, alcohol use, and exercise level.</p>
        {df_to_html_table(categorical[categorical["column"].isin(cat_cols)], max_rows=60, digits=2)}
      </section>

      <section id="preprocessing">
        <h2>Preprocessing and Feature Engineering</h2>
        <p class="section-lead">Preprocessing was implemented in a reusable sklearn pipeline so train-fitted transformations are applied consistently in modeling and deployment.</p>
        <h3>Schema Cleanup</h3>
        {df_to_html_table(rename_cleanup, digits=3)}
        <h3>Missing Value Strategy</h3>
        <ul>
          <li><b>BMI:</b> imputed using median values by <code>age_band</code> and <code>Gender</code> where stable, falling back to global median.</li>
          <li><b>Year_last_admitted:</b> converted into <code>was_admitted_before</code> and <code>years_since_last_admitted</code>. Missing values are coded as no prior admission with recency 0.</li>
          <li><b>Smoking status:</b> <code>Unknown</code> is retained as a valid business category, not forced to null.</li>
        </ul>
        <h3>Engineered Features</h3>
        <div>
          <span class="tag">age_band</span>
          <span class="tag">bmi_category</span>
          <span class="tag">cholesterol_midpoint</span>
          <span class="tag">was_admitted_before</span>
          <span class="tag">years_since_last_admitted</span>
          <span class="tag">any_major_disease_history</span>
          <span class="tag">weight_bmi_interaction</span>
          <span class="tag">steps_per_age</span>
          <span class="tag">medical_risk_score</span>
          <span class="tag">lifestyle_risk_score</span>
          <span class="tag">preventive_care_score</span>
          <span class="tag">age_bmi_interaction</span>
          <span class="tag">bmi_glucose_interaction</span>
          <span class="tag">smoking_alcohol_interaction</span>
          <span class="tag">risk_profile_segment</span>
        </div>
        <h3>Outlier Review</h3>
        <p>Target outliers were retained because high premium cases can be valid and business-relevant. BMI was capped only for extreme unrealistic values to improve stability.</p>
        {df_to_html_table(outliers, digits=3)}
        {figure("outlier_boxplots.png", "Outlier review for BMI, weight, glucose, daily steps, fat percentage, and insurance cost.")}
      </section>

      <section id="eda">
        <h2>EDA Insights</h2>
        <p class="section-lead">EDA was used to understand target behavior, missingness, feature relationships, and business segments before model training.</p>
        <div class="grid-2">
          {figure("target_distribution.png", "Target distribution: insurance_cost is mildly right-skewed.")}
          {figure("target_boxplot.png", "Target boxplot: no clearly invalid target values were removed.")}
          {figure("target_price_grid_frequency.png", "Target price-grid frequency across the 54 valid quote bands.")}
        </div>
        <h3>Numeric Relationships</h3>
        <div class="grid-2">
          {figure("correlation_heatmap.png", "Correlation heatmap for numeric and engineered variables.")}
          {figure("cost_vs_weight.png", "Insurance cost versus weight shows a very strong positive relationship.")}
          {figure("cost_vs_years_since_admitted.png", "Admission recency/history provides a useful risk signal.")}
          {figure("cost_vs_glucose.png", "Insurance cost versus average glucose level.")}
          {figure("cost_vs_steps.png", "Insurance cost versus daily average steps.")}
        </div>
        <h3>Segment-Level EDA</h3>
        <p>Group summaries show how premium distributions vary across lifestyle and risk-history segments. Several lifestyle categories have similar medians, but distribution shifts and model interactions still matter.</p>
        {"".join(f"<h4>{html.escape(name)}</h4>{df_to_html_table(tbl, digits=2)}" for name, tbl in group_tables)}
        <div class="grid-2">
          {figure("cost_by_smoking.png", "Insurance cost by smoking status.")}
          {figure("cost_by_exercise.png", "Insurance cost by exercise category.")}
          {figure("cost_by_alcohol.png", "Insurance cost by alcohol consumption.")}
          {figure("cost_by_adventure_sports.png", "Insurance cost by adventure sports flag.")}
          {figure("cost_by_disease_history.png", "Insurance cost by any major disease history flag.")}
          {figure("cost_by_other_coverage.png", "Insurance cost by other insurance coverage.")}
          {figure("cost_by_bmi_category.png", "Insurance cost by BMI category.")}
          {figure("disease_history_means.png", "Average cost by disease-history combination.")}
          {figure("cost_age_bmi_heatmap.png", "Average cost by age band and BMI category.")}
          {figure("cost_smoking_by_age_band.png", "Average cost by smoking status across age bands.")}
          {figure("cost_by_combined_risk_bands.png", "Average cost by engineered medical and lifestyle risk bands.")}
        </div>
        <div class="callout">
          <b>EDA takeaway:</b> Weight, admission history, other coverage, preventive checkups, and weight change became the strongest model drivers. Lifestyle features should still be monitored because they can matter through nonlinear interactions and segment-level policies.
        </div>
      </section>

      <section id="modeling">
        <h2>Modeling Approach and Performance</h2>
        <p class="section-lead">All models used the same 80/20 train-test split with <code>random_state=42</code>. Preprocessing was fitted only on training data inside a sklearn Pipeline.</p>
        <h3>Model Families Compared</h3>
        <ul>
          <li><b>Baseline:</b> DummyRegressor.</li>
          <li><b>Linear:</b> Linear Regression.</li>
          <li><b>Regularized:</b> Ridge.</li>
          <li><b>Tree and ensembles:</b> Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting.</li>
          <li><b>Optional boosters:</b> {html.escape(available_boosters)} were available in the active environment; unavailable packages are skipped safely.</li>
        </ul>
        <h3>Metrics Used</h3>
        <ul>
          <li><b>MAE:</b> average absolute premium error.</li>
          <li><b>RMSE:</b> penalizes larger pricing errors.</li>
          <li><b>R2:</b> share of target variance explained.</li>
          <li><b>MAPE/SMAPE:</b> business-friendly percentage-style error measures.</li>
          <li><b>Train-test RMSE gap:</b> checks overfitting risk.</li>
        </ul>
        <h3>Model Comparison</h3>
        {df_to_html_table(metrics, digits=3)}
        {figure("model_comparison_rmse.png", "Model comparison by test RMSE.")}
        <h3>Previous Run Comparison</h3>
        <p>This table is retained only as a previous-run comparison. It is not treated as an apples-to-apples improvement claim when the split or model set changed.</p>
        {df_to_html_table(previous_run_comparison, digits=4) if not previous_run_comparison.empty else "<p class='muted'>No previous-run comparison file was available.</p>"}
        <h3>Final Model Selection</h3>
        <div class="callout">
          The selected final model is <b>{metadata["final_model"]}</b>. It was selected with a parsimony/stability rule, not because it necessarily had the absolute lowest test RMSE. {html.escape(metadata.get("selection_reason", ""))}
        </div>
        {df_to_html_table(final_model_summary, digits=4) if not final_model_summary.empty else ""}
        <h3>Calibration and Quote-band Evaluation</h3>
        <p>Calibration and nearest-band rounding are reported honestly as deployment variants. Calibration improves some percentage/band metrics but does not automatically replace the raw prediction for MAE/RMSE-oriented analytics.</p>
        {df_to_html_table(price_grid_evaluation, digits=4) if not price_grid_evaluation.empty else ""}
        {df_to_html_table(calibration_comparison, digits=4) if not calibration_comparison.empty else ""}
        <div class="grid-2">
          {figure("calibration_curve.png", "Calibration curve comparing raw and calibrated predictions.")}
          {figure("residual_by_cost_decile_after_calibration.png", "Mean residual by actual-cost decile before and after calibration.")}
        </div>
        <div class="grid-2">
          {figure("predicted_vs_actual.png", "Predicted versus actual insurance cost for the final model.")}
          {figure("residuals.png", "Residuals versus predicted cost for the final model.")}
          {figure("error_by_cost_decile.png", "Prediction error by actual insurance-cost decile.")}
          {figure("segment_error_mae.png", "Highest segment-level prediction errors.")}
          {figure("segment_bias.png", "Largest segment-level prediction biases.")}
        </div>
        <h3>Final Model Parameters</h3>
        {df_to_html_table(final_params, digits=3)}
        <h3>Residual and Fairness Diagnostics</h3>
        <p>Segment diagnostics summarize prediction error and bias. Positive bias means predicted cost is higher than actual; negative bias means underprediction.</p>
        {df_to_html_table(residual_segments.sort_values("MAE", ascending=False).head(30), digits=3) if not residual_segments.empty else ""}
        {df_to_html_table(fairness_summary, digits=3) if not fairness_summary.empty else ""}
      </section>

      <section id="explainability">
        <h2>Explainability</h2>
        <p class="section-lead">Permutation importance was used as the model-agnostic explainability method. SHAP status: {html.escape(str(shap_status))}.</p>
        <p>Top model drivers: <b>{html.escape(top_feature_text)}</b>.</p>
        {df_to_html_table(importance, max_rows=30, digits=4)}
        <div class="grid-2">
          {figure("feature_importance.png", "Top model drivers by permutation importance.")}
          {figure("partial_dependence_top_features.png", "Partial dependence plots for top numeric drivers.")}
        </div>
        <div class="warning callout">
          <b>Interpretation caution:</b> The model finds <code>weight</code> as a dominant predictor in this dataset. That may reflect the underlying data-generation pattern or business process. In a real insurer setting, this should be reviewed for plausibility, fairness, and leakage before production use.
        </div>
      </section>

      <section id="deployment">
        <h2>Streamlit Deployment</h2>
        <p>The deployment app is implemented in <code>app.py</code>. It loads <code>outputs/models/final_model.pkl</code>, accepts applicant features through a form, displays raw predicted cost, nearest selected quote band, optional calibrated cost, Low / Medium / High risk category, and top model drivers.</p>
        <p class="muted">Deployment variant: analytics prediction = <code>{html.escape(str(deployment_variant.get("analytics_prediction", "raw_continuous")))}</code>; quote band = <code>{html.escape(str(deployment_variant.get("quote_band", "rounded_to_nearest_price_band")))}</code>; calibrated prediction = <code>{html.escape(str(deployment_variant.get("calibrated_prediction", "secondary_diagnostic")))}</code>.</p>
        {figure("streamlit_deployment_workflow.png", "Streamlit deployment workflow.")}
        <h3>Risk Category Thresholds</h3>
        <div class="stat-grid">
          {stat_card("Low to Medium", fmt_num(metadata["risk_thresholds"]["low_medium"], 0), "Prediction threshold")}
          {stat_card("Medium to High", fmt_num(metadata["risk_thresholds"]["medium_high"], 0), "Prediction threshold")}
          {stat_card("Train Rows", f"{metadata['train_rows']:,}", "Model training split")}
          {stat_card("Test Rows", f"{metadata['test_rows']:,}", "Final holdout split")}
        </div>
        <pre><code>streamlit run app.py</code></pre>
        <p class="muted">The app includes a disclaimer that the model supports pricing analysis and triage, not final underwriting decisions.</p>
      </section>

      <section id="recommendations">
        <h2>Business Recommendations</h2>
        <ol>
          <li><b>Use the model as pricing decision support.</b> It should guide quote review and risk triage, not automatically approve or reject policies.</li>
          <li><b>Manual review for extreme predictions.</b> Very high or unusual predicted premiums should be routed to an underwriting analyst before final quote approval.</li>
          <li><b>Wellness targeting.</b> Build wellness nudges around weight management, preventive checkups, and activity improvement for high-risk customer segments.</li>
          <li><b>Admission-history review.</b> Prior hospitalization and admission recency should trigger preventive-health and documentation checks.</li>
          <li><b>Governance on sensitive fields.</b> Gender and Location must be reviewed for fairness, legality, and proxy-discrimination risk before real-world use.</li>
          <li><b>Monitoring after deployment.</b> Track drift, segment-level error, quote conversion, and manual override rates.</li>
        </ol>
      </section>

      <section id="risks">
        <h2>Risks, Limitations, and Controls</h2>
        <div class="grid-2">
          <div>
            <h3>Limitations</h3>
            <ul>
              <li>Dataset provenance and production pricing rules are not specified.</li>
              <li>All predictors must be confirmed available at quote time.</li>
              <li>SHAP and optional booster availability depends on the installed runtime; the report records package status in model metadata.</li>
              <li>Segment-level fairness testing should be expanded before real production use.</li>
            </ul>
          </div>
          <div>
            <h3>Controls</h3>
            <ul>
              <li>Use reproducible sklearn Pipeline artifacts.</li>
              <li>Keep applicant_id out of modeling.</li>
              <li>Monitor prediction distributions and model drift.</li>
              <li>Retain manual review for outlier predictions.</li>
            </ul>
          </div>
        </div>
      </section>

      <section id="files">
        <h2>Files, Deliverables, and Reproducibility</h2>
        <p class="section-lead">The capstone solution is reproducible from the current folder.</p>
        <h3>Final Deliverables</h3>
        <div class="deliverables">
          {artifact_link("Milestone_1_Insurance_Price_Prediction.docx", "Milestone 1 report")}
          {artifact_link("Milestone_2_Insurance_Price_Prediction.docx", "Milestone 2 report")}
          {artifact_link("Insurance_Price_Prediction_Final_Presentation.pptx", "Final PowerPoint presentation")}
          {artifact_link("app.py", "Streamlit app")}
          {artifact_link("outputs/models/final_model.pkl", "Final model artifact")}
          {artifact_link("outputs/models/model_metrics.csv", "Model metrics CSV")}
          {artifact_link("outputs/models/feature_importance.csv", "Feature importance CSV")}
          {artifact_link("Insurance_Price_Prediction_Capstone_Project.zip", "Full project zip")}
        </div>
        <h3>Project Structure</h3>
        <pre><code>outputs/
  figures/
  tables/
  models/
  reports/
notebooks/
  01_milestone1_eda.ipynb
  02_milestone2_modeling.ipynb
app.py
insurance_modeling.py
run_all.py
requirements.txt
README.md</code></pre>
        <h3>Rebuild Command</h3>
        <pre><code>python run_all.py</code></pre>
        <p>The command regenerates EDA figures, summary tables, model artifacts, reports, presentation, notebooks, and Streamlit app dependencies.</p>
      </section>

      <div class="footer">
        Generated from local project artifacts in <code>{html.escape(str(ROOT))}</code>.
      </div>
    </main>
  </div>
</body>
</html>
"""

    OUTPUT.write_text(html_content, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

