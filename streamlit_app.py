"""Thin Streamlit demo for the insurance-cost model (capstone rubric).

Reuses the production prediction code in ``insurance.predict`` and the shared
field metadata in ``insurance.schemas`` so it stays consistent with the React
UI and the FastAPI service.

Run:  uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from insurance import config
from insurance.predict import load_bundle, predict
from insurance.schemas import FIELD_SPECS

st.set_page_config(page_title="Insurance Price Prediction", page_icon="💰", layout="centered")

st.title("💰 Insurance Price Prediction")
st.caption("Decision-support tool — estimates insurance cost from health & lifestyle inputs. Not a final underwriting decision.")

if not config.MODEL_PATH.exists():
    st.error("Model artifact not found. Run `uv run python -m insurance.train` first.")
    st.stop()

bundle = load_bundle()
st.sidebar.header("Model")
st.sidebar.write(f"**{bundle.get('model_name', 'model')}**")
m = bundle.get("metrics", {})
if m:
    st.sidebar.metric("Test R²", f"{m.get('test_R2', 0):.3f}")
    st.sidebar.metric("Test MAE", f"{m.get('test_MAE', 0):,.0f}")

with st.form("applicant"):
    st.subheader("Applicant details")
    values: dict = {}
    cols = st.columns(2)
    for i, spec in enumerate(FIELD_SPECS):
        col = cols[i % 2]
        name, label = spec["name"], spec["label"]
        if spec["kind"] == "select":
            values[name] = col.selectbox(label, spec["options"],
                                         index=spec["options"].index(spec["default"]))
        elif spec["kind"] == "binary":
            values[name] = int(col.selectbox(label, spec["options"],
                                             index=spec["options"].index(spec["default"])))
        else:  # number
            default = spec.get("default")
            if default is None:  # optional numeric (e.g. Year_last_admitted)
                raw = col.text_input(label, value="")
                values[name] = float(raw) if raw.strip() else None
            else:
                step = spec.get("step", 1)
                values[name] = col.number_input(
                    label, min_value=float(spec["min"]), max_value=float(spec["max"]),
                    value=float(default), step=float(step),
                )
    submitted = st.form_submit_button("Predict insurance cost", use_container_width=True)

if submitted:
    result = predict(values, bundle=bundle)[0]
    cost = result["predicted_cost"]
    risk = result["risk_category"]
    color = {"Low": "green", "Medium": "orange", "High": "red"}[risk]
    c1, c2 = st.columns(2)
    c1.metric("Predicted insurance cost", f"₹ {cost:,.0f}")
    c2.markdown(f"### Risk band: :{color}[{risk}]")

    drivers = bundle.get("feature_importance", [])
    if drivers:
        st.subheader("Top model drivers")
        st.dataframe(
            [{"feature": d["feature"], "importance": round(d["importance_mean"], 2)} for d in drivers],
            hide_index=True, use_container_width=True,
        )
