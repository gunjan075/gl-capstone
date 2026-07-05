Insurance Price Prediction, Milestone 1 workflow for a capstone data report.

The project predicts insurance_cost from applicant health, lifestyle, habit, demographic, and insurance history fields.
Milestone 1 is not model training. It prepares the evidence base for Milestone 2.

Key analytical steps:
1. Load the raw Insurance Data.csv file with 25,000 applicant rows and 24 original columns.
2. Audit data quality: duplicate rows, applicant_id uniqueness, column names, missing values, and variable types.
3. Interpret the target as a discrete quote grid: 54 valid insurance_cost bands with a step size of 1,234.
4. Analyze missingness: BMI has limited missingness; Year_last_admitted has structural missingness and should become admission-status and recency features instead of being dropped.
5. Run univariate EDA for every original variable, including numeric, categorical, ordinal, binary, identifier, and target fields.
6. Run bivariate and segmented EDA against insurance_cost to identify observed signals.
7. Correctly summarize the strongest observed signals: weight, admission recency or Year_last_admitted, covered_by_any_other_company, regular_checkup_last_year, weight_change_in_last_one_year, and adventure_sports.
8. Treat weaker marginal signals honestly: smoking, alcohol, exercise, age, BMI, glucose, cholesterol, and disease-history flags are useful context but not the strongest standalone drivers in this dataset.
9. Create preprocessing decisions: remove applicant_id from modeling, fix typos, engineer target-grid features, impute and flag BMI, create admission-history fields, convert cholesterol range to midpoint, and build interpretable risk/context features.
10. Handoff to Milestone 2: target-band stratified train/test split, model comparison, calibration and quote-band rounding evaluation, explainability, and Streamlit deployment.

Visual intent:
Create a clean academic workflow diagram for a Word report. Use a left-to-right flow with six stages and these exact labels:
1. Raw Dataset
   Small text: 25,000 rows, 24 columns
2. Data Quality Audit
   Small text: duplicates, missingness, variable types
3. Target Grid Discovery
   Small text: insurance_cost quote bands
4. EDA Signal Review
   Small text: univariate, bivariate, segmented EDA
5. Preprocessing Decisions
   Small text: impute BMI, create admission recency, engineer risk features
6. Modeling-Ready Dataset
   Small text: handoff to Milestone 2 split

Include exactly two callouts:
- 54 quote bands, step 1,234
- applicant_id excluded

Use professional colors, readable labels, and simple icons or abstract shapes. Do not include screenshots, code, formulas, decorative stock imagery, italic asterisk styling, or misspelled words. The words "formats" and "fomats" must not appear. The diagram should look like a publication-ready process figure for a data science capstone report.
