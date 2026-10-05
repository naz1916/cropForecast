import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

EXPORT = Path(__file__).parent / "crop_forecast_export"

WEATHER_COLS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "rain_sum", "precipitation_hours", "sunshine_duration",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "wind_speed_10m_max", "wind_gusts_10m_max", "rain_normal", "temp_normal",
    "rainfall_deviation_pct", "temperature_anomaly_c", "oni_index",
    "temp_anomaly_rolling", "rain_anomaly_rolling", "climate_stress",
]
ENSO = ["El Niño", "La Niña", "Neutral"]  # El Niño = baseline (no dummy column)
RISK_COLORS = {"Stable": "#2e8b57", "Risk-prone": "#e6a117",
               "Declining": "#d9534f", "High": "#8e44ad"}

st.set_page_config(page_title="cropPHorecast", page_icon="🌾", layout="wide")


@st.cache_resource
def load_assets():
    reg = joblib.load(EXPORT / "regressor.joblib")
    clf = joblib.load(EXPORT / "classifier.joblib")
    meta = json.loads((EXPORT / "meta.json").read_text())
    ref = pd.read_csv(EXPORT / "reference.csv.gz")
    ref["t"] = ref["year"] * 4 + ref["quarter_num"]
    return reg, clf, meta, ref


def build_row(meta, province, crop, year, qnum, lags, weather, enso):
    """Return a 1-row DataFrame in the exact training column order."""
    row = dict.fromkeys(meta["reg_feature_cols"], 0)
    row.update(weather)
    row.update({
        "year": year, "quarter_num": qnum, "time_index": year * 4 + qnum,
        "production_lag_1": lags[1], "production_lag_2": lags[2],
        "production_lag_4": lags[4],
        "crop_encoded": meta["crop_classes"].index(crop),
        "enso_phase_La Niña": int(enso == "La Niña"),
        "enso_phase_Neutral": int(enso == "Neutral"),
    })
    for col in (f"province_{province}", f"crop_group_{meta['crop_to_group'][crop]}"):
        if col in row:
            row[col] = 1
    return pd.DataFrame([row])[meta["reg_feature_cols"]]


reg, clf, meta, ref = load_assets()

st.title("🌾 cropPHorecast")
st.caption("Quarterly crop production forecast and risk classification for Philippine provinces.")

# ---------------- Sidebar: selection ----------------
with st.sidebar:
    st.header("Scenario")
    province = st.selectbox("Province", meta["provinces"])
    crops = sorted(ref.loc[ref["province"] == province, "crop"].unique())
    crop = st.selectbox("Crop", crops)

hist = (ref[(ref["province"] == province) & (ref["crop"] == crop)]
        .sort_values("t").drop_duplicates("t", keep="last"))
last_t = int(hist["t"].max())
default_t = last_t + 1
d_year, d_q = divmod(default_t - 1, 4)
d_q += 1

with st.sidebar:
    year = st.number_input("Target year", 2010, 2035, int(d_year))
    qnum = st.selectbox("Target quarter", [1, 2, 3, 4], index=int(d_q) - 1)
    st.caption(f"Latest data for this crop: {(last_t - 1) // 4} Q{(last_t - 1) % 4 + 1}")

target_t = int(year) * 4 + int(qnum)

# ---------------- Lags from history ----------------
prod_by_t = hist.set_index("t")["production"]
lags, missing = {}, []
for k in (1, 2, 4):
    v = prod_by_t.get(target_t - k, np.nan)
    if np.isnan(v):
        missing.append(k)
        v = 0.0
    lags[k] = float(v)

# ---------------- Default weather: same quarter, most recent year ----------------
same_q = hist[hist["quarter_num"] == qnum]
src = same_q.iloc[-1] if len(same_q) else hist.iloc[-1]
defaults = {c: float(src[c]) for c in WEATHER_COLS}
default_enso = ("La Niña" if src.get("enso_phase_La Niña", False)
                else "Neutral" if src.get("enso_phase_Neutral", False) else "El Niño")

st.subheader(f"{crop.title()} — {province.title()} — {int(year)} Q{int(qnum)}")
if missing:
    st.warning(f"No history for lag(s) {missing} quarter(s) back — using 0. "
               "Pick a target quarter closer to the available data for a reliable forecast.")

with st.expander("Inputs (pre-filled from history — edit to run what-if scenarios)", expanded=False):
    c1, c2, c3 = st.columns(3)
    lags[1] = c1.number_input("Production, 1 quarter ago", value=lags[1])
    lags[2] = c2.number_input("Production, 2 quarters ago", value=lags[2])
    lags[4] = c3.number_input("Production, 4 quarters ago", value=lags[4])
    enso = st.radio("ENSO phase", ENSO, index=ENSO.index(default_enso), horizontal=True)
    weather = {}
    cols = st.columns(3)
    for i, c in enumerate(WEATHER_COLS):
        weather[c] = cols[i % 3].number_input(c, value=defaults[c], format="%.3f")

# ---------------- Predict ----------------
X = build_row(meta, province, crop, int(year), int(qnum), lags, weather, enso)
prod_pred = max(float(reg.predict(X)[0]), 0.0)
proba = clf.predict_proba(X[meta["clf_feature_cols"]])[0]
risk_pred = meta["risk_classes"][int(np.argmax(proba))]

m1, m2 = st.columns(2)
m1.metric("Predicted production", f"{prod_pred:,.0f}",
          delta=f"{prod_pred - lags[4]:,.0f} vs same quarter last year" if lags[4] else None)
m2.metric("Predicted risk class", risk_pred)

left, right = st.columns(2)
with left:
    st.markdown("**Production history + forecast**")
    h = hist.tail(16)[["t", "production"]].copy()
    h["label"] = h["t"].apply(lambda t: f"{(t - 1) // 4}Q{(t - 1) % 4 + 1}")
    chart = pd.concat([
        h[["label", "production"]].rename(columns={"production": "Actual"}),
        pd.DataFrame({"label": [f"{int(year)}Q{int(qnum)}"], "Forecast": [prod_pred]}),
    ]).set_index("label")
    st.line_chart(chart)
with right:
    st.markdown("**Risk class probabilities**")
    st.bar_chart(pd.Series(proba, index=meta["risk_classes"], name="probability"))

with st.expander("About these models"):
    rm, cm = meta["reg_metrics"], meta["clf_metrics"]
    st.markdown(
        f"- **Regression:** {meta['regressor_name']} — holdout RMSE {rm['RMSE']:,.0f}, "
        f"MAE {rm['MAE']:,.0f}, R² {rm['R2']:.3f}\n"
        f"- **Classification:** {meta['classifier_name']} — holdout accuracy {cm['Accuracy']:.3f}, "
        f"weighted F1 {cm['F1-Score']:.3f}\n"
        f"- Trained on {meta['train_years'][0]}–{meta['train_years'][1]}; evaluated on a forward "
        "time-based holdout (2023–2025).\n"
        "- The classifier is weak on the *Declining* and *High* classes (recall below 10% in the "
        "notebook's Random Forest run), so treat those probabilities as indicative, not definitive."
    )
