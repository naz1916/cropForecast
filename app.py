"""CropForecast PH - Streamlit app: A. Forecast  B. Nowcast  C. Next-quarter risk label."""
from __future__ import annotations

import os
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="CropForecast PH", page_icon="🌾", layout="wide")

CSV_NAME = "crop_trend_master_2025q4.csv"
# Google Drive copy of the dataset (file must be shared as "Anyone with the link"):
# https://drive.google.com/file/d/1qh97JxqiAmqNW8rSE-LTgX_7joLi85NQ/view?usp=sharing
GDRIVE_FILE_ID = "1qh97JxqiAmqNW8rSE-LTgX_7joLi85NQ"

# ------------------------------------------------------------------ feature guide (units + descriptions)
# name: (label, unit, description, kind)   kind: input / derived
GUIDE = {
    "province": ("Province", "-", "Province of the production series (PSA naming, uppercase).", "input"),
    "crop": ("Crop", "-", "Crop name from the harmonized PSA crop table (89 crops).", "input"),
    "crop_group": ("Crop group", "-", "PALAY, CORN or OTHER. Set automatically from the crop.", "derived"),
    "year": ("Target year", "calendar year", "Year of the quarter being predicted.", "input"),
    "quarter": ("Target quarter", "Q1-Q4", "Quarter being predicted (Q1 = Jan-Mar ... Q4 = Oct-Dec).", "input"),
    "production": ("Production this quarter", "metric tons", "Production of the current quarter (t). Used by the next-quarter label model.", "input"),
    "production_lag_1": ("Production 1 quarter earlier", "metric tons", "Production of the previous quarter, same province and crop. 0 = not recorded.", "input"),
    "production_lag_2": ("Production 2 quarters earlier", "metric tons", "Production two quarters earlier. 0 = not recorded.", "input"),
    "production_lag_3": ("Production 3 quarters earlier", "metric tons", "Production three quarters earlier (needed by the label model). 0 = not recorded.", "input"),
    "production_lag_4": ("Production 4 quarters earlier", "metric tons", "Same quarter last year: the year-over-year baseline. 0 = not recorded.", "input"),
    "growth_rate_yoy": ("Year-over-year growth", "ratio (0.10 = +10%)", "(production - production_lag_4) / production_lag_4. Computed for you.", "derived"),
    "temperature_2m_mean": ("Mean air temperature", "°C", "Quarter mean of the daily mean air temperature at 2 m.", "input"),
    "temperature_2m_max": ("Max air temperature", "°C", "Quarter mean of the daily maximum air temperature at 2 m.", "input"),
    "temperature_2m_min": ("Min air temperature", "°C", "Quarter mean of the daily minimum air temperature at 2 m.", "input"),
    "precipitation_sum": ("Total precipitation", "mm", "Quarter sum of daily total precipitation (rain + other).", "input"),
    "rain_sum": ("Total rain", "mm", "Quarter sum of daily rain only.", "input"),
    "precipitation_hours": ("Hours with rain", "hours", "Quarter total of hours with precipitation.", "input"),
    "sunshine_duration": ("Sunshine duration", "hours (stored in seconds)", "Quarter total sunshine. You enter hours; the app converts to seconds like the dataset.", "input"),
    "shortwave_radiation_sum": ("Shortwave solar radiation", "MJ/m²", "Quarter sum of daily shortwave radiation reaching the surface.", "input"),
    "et0_fao_evapotranspiration": ("Reference evapotranspiration (ET0)", "mm", "Quarter sum of FAO-56 reference evapotranspiration: water the atmosphere can pull from a reference crop.", "input"),
    "wind_speed_10m_max": ("Max wind speed", "km/h", "Quarter mean of the daily maximum wind speed at 10 m.", "input"),
    "wind_gusts_10m_max": ("Max wind gusts", "km/h", "Quarter mean of the daily maximum wind gust at 10 m.", "input"),
    "rain_normal": ("Rainfall normal", "mm", "Long-term normal quarterly rainfall for the province; baseline for the deviation.", "input"),
    "temp_normal": ("Temperature normal", "°C", "Long-term normal quarterly temperature for the province; baseline for the anomaly.", "input"),
    "oni_index": ("ONI (ENSO index)", "°C", "Oceanic Niño Index. ≥ +0.5 El Niño, ≤ -0.5 La Niña, otherwise Neutral.", "input"),
    "rain_anomaly_rolling": ("Rainfall anomaly, 4-qtr rolling mean", "% deviation", "Mean of rainfall_deviation_pct over the last 4 quarters including this one.", "input"),
    "temp_anomaly_rolling": ("Temperature anomaly, 4-qtr rolling mean", "°C", "Mean of temperature_anomaly_c over the last 4 quarters including this one.", "input"),
    "rainfall_deviation_pct": ("Rainfall deviation", "%", "(precipitation_sum - rain_normal) / rain_normal x 100. Computed for you.", "derived"),
    "temperature_anomaly_c": ("Temperature anomaly", "°C", "temperature_2m_mean - temp_normal. Computed for you.", "derived"),
    "enso_phase": ("ENSO phase (one-hot flags)", "El Niño / Neutral / La Niña", "Derived from ONI and encoded as two flags (El Niño is the omitted reference).", "derived"),
    "climate_stress": ("Climate stress", "index (% + °C)", "|rain_anomaly_rolling| + |temp_anomaly_rolling|. Computed for you.", "derived"),
    "risk_label": ("Risk label", "class", "Stable / Risk-prone / High / Declining: a rule-based heuristic, not an official or agronomic standard.", "derived"),
}


# ------------------------------------------------------------------ Arrow-free display helpers
# st.table / st.dataframe / st.line_chart convert pandas objects to Apache Arrow, which raised
# ArrowNotImplementedError on some pandas/pyarrow combinations. These helpers use plain markdown / JSON instead.
def show_table(df: pd.DataFrame, nd: int = 3):
    def cell(v):
        if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{v:,.{nd}f}"
        return str(v).replace("|", "\\|")
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in zip(df.index, df.itertuples(index=False)):
        lines.append("| " + " | ".join([cell(idx)] + [cell(v) for v in row]) + " |")
    st.markdown("\n".join(lines))


def line_chart(records: list[dict], ytitle: str):
    ch = (alt.Chart(alt.Data(values=records)).mark_line(point=True)
          .encode(x=alt.X("period:N", sort=None, title=None), y=alt.Y("value:Q", title=ytitle),
                  color=alt.Color("series:N", title=None), tooltip=["period:N", "series:N", "value:Q"]))
    st.altair_chart(ch, use_container_width=True)


def bar_chart(labels: list[str], values: list[float]):
    recs = [{"label": l, "probability": float(v)} for l, v in zip(labels, values)]
    ch = (alt.Chart(alt.Data(values=recs)).mark_bar()
          .encode(x=alt.X("label:N", sort=None, title=None), y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1])),
                  tooltip=["label:N", alt.Tooltip("probability:Q", format=".1%")]))
    st.altair_chart(ch, use_container_width=True)


@st.cache_resource(show_spinner="Loading dataset …")
def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def get_data(key: str, _raw: pd.DataFrame) -> core.Data:
    return core.Data(_raw)


@st.cache_resource(show_spinner=False)
def get_bundle(key: str, task: str, model_name: str, _data: core.Data) -> dict:
    if task == "forecast":
        return core.train_regressor(_data, model_name, core.REG_FORECAST)
    if task == "nowcast":
        return core.train_regressor(_data, model_name, core.REG_NOWCAST)
    return core.train_classifier(_data, model_name)


def fmt(x, nd=1):
    return f"{x:,.{nd}f}"


def bundle_with_spinner(task, model_name):
    slow = {"forecast": "a few seconds", "nowcast": "10-60 s", "label": "up to ~1-2 min"}[task]
    with st.spinner(f"Training {model_name} for the {task} task on the full dataset ({slow}, first run only) …"):
        return get_bundle(DATA_KEY, task, model_name, DATA)


# ------------------------------------------------------------------ sidebar: data + model
st.sidebar.title("🌾 CropForecast PH")
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_NAME)


def get_drive_id():
    """Drive file ID: Streamlit secret / env variable if set (override), otherwise the ID built into this file."""
    try:
        v = st.secrets.get("GDRIVE_FILE_ID", None)
    except Exception:          # no secrets file
        v = None
    return v or os.environ.get("GDRIVE_FILE_ID") or GDRIVE_FILE_ID


@st.cache_resource(show_spinner="Downloading dataset from Google Drive (first start only) …")
def fetch_from_drive(file_id: str) -> str:
    import tempfile
    import gdown
    out = os.path.join(tempfile.gettempdir(), CSV_NAME)
    if not os.path.exists(out):
        gdown.download(id=file_id, output=out, quiet=True)
    if not os.path.exists(out):
        raise FileNotFoundError("Google Drive download failed: check that the file is shared as 'Anyone with the link'.")
    return out


if os.path.exists(here):                            # 1) CSV next to app.py (local use)
    DATA_KEY = f"file-{os.path.getmtime(here)}"
    raw = load_raw(here)
else:                                               # 2) Google Drive
    drive_id = get_drive_id()
    try:
        path = fetch_from_drive(drive_id)
    except Exception as e:
        st.title("🌾 CropForecast PH")
        st.error(f"Could not get the dataset from Google Drive: {e}")
        st.stop()
    DATA_KEY = f"drive-{drive_id}"
    raw = load_raw(path)
try:
    DATA = get_data(DATA_KEY, raw)
except KeyError as e:
    st.error(str(e)); st.stop()

models = core.available_models()
model_name = st.sidebar.selectbox("Model", models, help="Trained on the full dataset the first time it is used, then cached. "
                                  "LightGBM / XGBoost / CatBoost appear only if installed (see requirements.txt).")
st.sidebar.caption(f"{len(DATA.df):,} rows · {len(DATA.provinces)} provinces · {len(DATA.crops)} crops · "
                   f"{int(DATA.df.year.min())}-{DATA.last_year}")
st.sidebar.caption("Production is province-level and in **metric tons per quarter**.")

st.title("🌾 Philippine crop production: forecast, nowcast & risk label")
tab_f, tab_n, tab_c, tab_g = st.tabs(["A · Forecast", "B · Nowcast", "C · Next-quarter risk label", "Feature guide"])


# ------------------------------------------------------------------ shared widgets
def inp_label(key, unit_override=None):
    lab, unit, _, _ = GUIDE[key]
    return f"{lab} ({unit_override or unit})"


def pick_series(prefix):
    c1, c2 = st.columns(2)
    prov = c1.selectbox("Province", DATA.provinces, key=f"{prefix}_prov", help=GUIDE["province"][2])
    crop = c2.selectbox("Crop", DATA.crops_in(prov), key=f"{prefix}_crop", help=GUIDE["crop"][2])
    return prov, crop


def num(key, value, prefix, tag, unit_override=None, step=None, minv=None, maxv=None, fmt_="%.2f", col=None):
    """number_input whose widget key changes with the selected series so defaults refresh."""
    rng = DATA.feature_range(key) if key in DATA.df.columns else (np.nan, np.nan)
    hint = GUIDE[key][2] + ("" if np.isnan(rng[0]) else f"  Typical range in the data: {rng[0]:,.1f} to {rng[1]:,.1f}.")
    kw = dict(label=inp_label(key, unit_override), value=float(value), help=hint, format=fmt_, key=f"{prefix}_{key}_{tag}")
    kw["step"] = step if step is not None else 0.1
    if minv is not None: kw["min_value"] = minv
    if maxv is not None: kw["max_value"] = maxv
    return (col or st).number_input(**kw)


def lag_inputs(prefix, prov, crop, target_tix, lag_list):
    """Auto-fill lags from the dataset; user may overwrite. Returns dict + info about missing history."""
    s = DATA.series(prov, crop)
    tag = f"{prov}|{crop}"
    cols = st.columns(len(lag_list)); out = {}; missing = []
    for c, (name, k) in zip(cols, lag_list):
        v = s.get(target_tix - k, np.nan)
        if np.isnan(v): missing.append(k); v = 0.0
        out[name] = num(name, v, prefix, tag, minv=0.0, step=1.0, fmt_="%.3f", col=c)
    if missing:
        st.caption(f"⚠️ No record in the dataset for lag(s) {missing}: filled with 0 (the dataset's own convention for missing history). Overwrite if you know the value.")
    return out


def weather_inputs(prefix, prov, qn):
    """All weather / ENSO / rolling inputs; defaults = province-quarter medians."""
    clim = DATA.climatology(prov, qn); tag = f"{prov}|{qn}"
    g = lambda k: 0.0 if pd.isna(clim.get(k, np.nan)) else float(clim[k])
    w = {}
    with st.expander("🌡️ Temperature", expanded=True):
        c = st.columns(4)
        w["temperature_2m_mean"] = num("temperature_2m_mean", g("temperature_2m_mean"), prefix, tag, col=c[0])
        w["temperature_2m_max"] = num("temperature_2m_max", g("temperature_2m_max"), prefix, tag, col=c[1])
        w["temperature_2m_min"] = num("temperature_2m_min", g("temperature_2m_min"), prefix, tag, col=c[2])
        w["temp_normal"] = num("temp_normal", g("temp_normal"), prefix, tag, col=c[3])
    with st.expander("🌧️ Rainfall", expanded=True):
        c = st.columns(4)
        w["precipitation_sum"] = num("precipitation_sum", g("precipitation_sum"), prefix, tag, minv=0.0, col=c[0])
        w["rain_sum"] = num("rain_sum", g("rain_sum"), prefix, tag, minv=0.0, col=c[1])
        w["precipitation_hours"] = num("precipitation_hours", g("precipitation_hours"), prefix, tag, minv=0.0, col=c[2])
        w["rain_normal"] = num("rain_normal", g("rain_normal"), prefix, tag, minv=0.0, col=c[3])
    with st.expander("☀️ Sun, evaporation & wind", expanded=False):
        c = st.columns(5)
        sun_h = num("sunshine_duration", g("sunshine_duration") / 3600, prefix, tag, minv=0.0, col=c[0])
        w["sunshine_duration"] = sun_h * 3600
        w["shortwave_radiation_sum"] = num("shortwave_radiation_sum", g("shortwave_radiation_sum"), prefix, tag, minv=0.0, col=c[1])
        w["et0_fao_evapotranspiration"] = num("et0_fao_evapotranspiration", g("et0_fao_evapotranspiration"), prefix, tag, minv=0.0, col=c[2])
        w["wind_speed_10m_max"] = num("wind_speed_10m_max", g("wind_speed_10m_max"), prefix, tag, minv=0.0, col=c[3])
        w["wind_gusts_10m_max"] = num("wind_gusts_10m_max", g("wind_gusts_10m_max"), prefix, tag, minv=0.0, col=c[4])
    with st.expander("🌊 ENSO & rolling anomalies", expanded=True):
        c = st.columns(3)
        w["oni_index"] = num("oni_index", g("oni_index"), prefix, tag, step=0.1, fmt_="%.2f", col=c[0])
        rain_roll = num("rain_anomaly_rolling", g("rain_anomaly_rolling"), prefix, tag, col=c[1])
        temp_roll = num("temp_anomaly_rolling", g("temp_anomaly_rolling"), prefix, tag, col=c[2])
    full = core.derive_weather_features(w, rain_roll, temp_roll)
    d = st.columns(5)
    d[0].metric("Rainfall deviation (%)", f"{full['rainfall_deviation_pct']:+.1f}", help=GUIDE["rainfall_deviation_pct"][2])
    d[1].metric("Temp. anomaly (°C)", f"{full['temperature_anomaly_c']:+.2f}", help=GUIDE["temperature_anomaly_c"][2])
    d[2].metric("ENSO phase", core.enso_from_oni(full["oni_index"]), help=GUIDE["enso_phase"][2])
    d[3].metric("Climate stress", f"{full['climate_stress']:.2f}", help=GUIDE["climate_stress"][2])
    d[4].metric("High-stress threshold (P75)", f"{DATA.p75_stress:.2f}", help="75th percentile of climate_stress in the dataset; used by the risk rule.")
    st.caption("Defaults are the median of this province and calendar quarter in 2010-2025: replace them with observed (or scenario) values.")
    return full


def base_row(prov, crop):
    return dict(province_code=DATA.prov_code[prov], crop_code=DATA.crop_code[crop],
                group_code=DATA.group_code[DATA.crop_group_of[crop]])


def history_chart(prov, crop, pred_tix, pred_val, n=16):
    s = DATA.series(prov, crop)
    s = s[s.index < pred_tix].tail(n)
    name = lambda t: f"{core.tix_to_yq(int(t))[0]} Q{core.tix_to_yq(int(t))[1]}"
    recs = [{"period": name(t), "series": "Observed", "value": float(v)} for t, v in zip(s.index, s.values)]
    if len(s):                                                          # connect the predicted line to the last observation
        recs.append({"period": name(s.index[-1]), "series": "Predicted", "value": float(s.values[-1])})
    recs.append({"period": name(pred_tix), "series": "Predicted", "value": float(pred_val)})
    line_chart(recs, "metric tons")


def target_picker(prefix, prov, crop):
    last = DATA.latest_tix(prov, crop)
    ty, tq = core.tix_to_yq(last + 1)
    st.caption(f"Latest observation for this series: {core.tix_to_yq(last)[0]} Q{core.tix_to_yq(last)[1]}. Default target = the next quarter.")
    c1, c2 = st.columns(2)
    year = c1.number_input(inp_label("year"), 2010, 2100, ty, 1, key=f"{prefix}_year_{prov}|{crop}", help=GUIDE["year"][2])
    quarter = c2.selectbox(inp_label("quarter"), [1, 2, 3, 4], index=tq - 1, key=f"{prefix}_q_{prov}|{crop}",
                           format_func=lambda q: f"Q{q}", help=GUIDE["quarter"][2])
    return int(year), int(quarter)


def reg_performance(b):
    m = b["metrics"]
    with st.expander(f"How good is this model? (hold-out test on {m['test_year']}, trained on earlier years only)"):
        show_table(pd.DataFrame({"MAE (metric tons)": [m["MAE_model"], m["MAE_naive"], m["MAE_seasonal_naive"]]},
                                index=[f"{b['name']}", "Naive: same as last quarter", "Seasonal naive: same quarter last year"]), nd=1)
        st.caption(f"{m['n_test']:,} test rows · model WAPE (volume-weighted % error) = {m['WAPE_model']:.1f}%. "
                   "MAE is dominated by large crops (palay, corn, banana…), so the miss for small crops is much smaller in absolute terms. "
                   "Compare the model with the seasonal-naive row: if it is not lower, a simple 'same as last year' is just as good.")


# ================================================================== A. FORECAST
with tab_f:
    st.subheader("A · Forecast: production of a future quarter (lags only)")
    st.markdown("Predicts **production (metric tons)** for a target quarter using only what is known **before** the quarter starts: "
                "province, crop, calendar and past production (lags 1, 2, 4). No weather is used.")
    prov, crop = pick_series("f")
    year, quarter = target_picker("f", prov, crop)
    tix = year * 4 + quarter
    st.markdown("**Past production** (auto-filled from the dataset, editable)")
    lags = lag_inputs("f", prov, crop, tix, [("production_lag_1", 1), ("production_lag_2", 2), ("production_lag_4", 4)])
    if st.button("Forecast production", type="primary", key="f_btn"):
        b = bundle_with_spinner("forecast", model_name)
        row = dict(base_row(prov, crop), qn=quarter, year=year, tix=tix, **lags)
        pred = core.predict_production(b, row)
        c = st.columns(4)
        c[0].metric(f"Forecast {year} Q{quarter}", f"{fmt(pred, 2)} t")
        c[1].metric("Naive (last quarter)", f"{fmt(lags['production_lag_1'], 2)} t")
        c[2].metric("Seasonal naive (same qtr last yr)", f"{fmt(lags['production_lag_4'], 2)} t")
        c[3].metric("Mean of lags 1, 2, 4", f"{fmt((lags['production_lag_1'] + lags['production_lag_2'] + lags['production_lag_4']) / 3, 2)} t")
        history_chart(prov, crop, tix, pred)
        reg_performance(b)
        if tix > DATA.latest_tix(prov, crop) + 1:
            st.warning("The target is more than one quarter beyond the latest data. Lags 1 and 2 only make sense if you typed realistic values; the further ahead, the less reliable.")


# ================================================================== B. NOWCAST
with tab_n:
    st.subheader("B · Nowcast: production of a quarter using that quarter's weather")
    st.markdown("Same target as the forecast, but the model **also sees the weather and ENSO of the predicted quarter** "
                "(temperature, rainfall, radiation, wind, ONI, rolling anomalies). Use it **during or after** the quarter, "
                "or to run **what-if scenarios** (e.g. drought vs. wet). It is not a forecast, because those weather values are unknown in advance.")
    prov, crop = pick_series("n")
    year, quarter = target_picker("n", prov, crop)
    tix = year * 4 + quarter
    st.markdown("**Past production** (auto-filled, editable)")
    lags = lag_inputs("n", prov, crop, tix, [("production_lag_1", 1), ("production_lag_2", 2), ("production_lag_4", 4)])
    st.markdown("**Weather of the target quarter**")
    wx = weather_inputs("n", prov, quarter)
    if st.button("Nowcast production", type="primary", key="n_btn"):
        bn = bundle_with_spinner("nowcast", model_name)
        bf = bundle_with_spinner("forecast", model_name)
        row = dict(base_row(prov, crop), qn=quarter, year=year, tix=tix, **lags, **wx)
        pred_n = core.predict_production(bn, row)
        pred_f = core.predict_production(bf, row)
        c = st.columns(3)
        c[0].metric(f"Nowcast {year} Q{quarter}", f"{fmt(pred_n, 2)} t")
        c[1].metric("Forecast (lags only) for comparison", f"{fmt(pred_f, 2)} t")
        c[2].metric("Effect of weather inputs", f"{pred_n - pred_f:+,.2f} t")
        history_chart(prov, crop, tix, pred_n)
        reg_performance(bn)
        st.caption("In the hold-out test, adding same-quarter weather changes accuracy only slightly: production lags carry most of the signal. "
                   "Treat the 'effect of weather' number as indicative, not causal.")


# ================================================================== C. NEXT-QUARTER LABEL
with tab_c:
    st.subheader("C · Next-quarter crop risk label")
    st.markdown("Given the situation at the **end of the current quarter (t)**, predicts the heuristic **risk label of quarter t+1**: "
                "`Stable`, `Risk-prone`, `High` (strong growth, low stress) or `Declining` (≥10% YoY drop).")
    prov, crop = pick_series("c")
    last = DATA.latest_tix(prov, crop)
    cy, cq = core.tix_to_yq(last)
    c1, c2 = st.columns(2)
    cur_year = c1.number_input("Current year (t)", 2010, 2100, cy, 1, key=f"c_year_{prov}|{crop}")
    cur_q = c2.selectbox("Current quarter (t)", [1, 2, 3, 4], index=cq - 1, key=f"c_q_{prov}|{crop}", format_func=lambda q: f"Q{q}")
    cur_tix = int(cur_year) * 4 + int(cur_q)
    ty, tq = core.tix_to_yq(cur_tix + 1)
    st.info(f"Prediction target: **{ty} Q{tq}**")

    b_c = None
    # ---- always-visible disclaimer with live numbers
    disc = st.container()

    s = DATA.series(prov, crop); tag = f"{prov}|{crop}|{cur_tix}"
    st.markdown("**Production** (auto-filled from the dataset, editable)")
    cols = st.columns(5); pv = {}
    for col, (name, k) in zip(cols, [("production", 0), ("production_lag_1", 1), ("production_lag_2", 2),
                                       ("production_lag_3", 3), ("production_lag_4", 4)]):
        v = s.get(cur_tix - k, 0.0)
        pv[name] = num(name, 0.0 if pd.isna(v) else v, "c", tag, minv=0.0, step=1.0, fmt_="%.3f", col=col)
    growth = (pv["production"] - pv["production_lag_4"]) / pv["production_lag_4"] if pv["production_lag_4"] > 0 else 0.0
    st.metric("Year-over-year growth (computed)", f"{growth * 100:+.1f}%", help=GUIDE["growth_rate_yoy"][2])
    st.markdown("**Weather of the current quarter (t)**")
    wx = weather_inputs("c", prov, int(cur_q))
    st.markdown("**Current risk label (t)**")
    rl = rule_default = core.rule_label(growth, wx["climate_stress"], wx["oni_index"], DATA.p75_stress)
    mode = st.radio("Source", ["Derive from the README rule", "Choose manually"], horizontal=True, key=f"c_mode_{tag}")
    if mode == "Choose manually":
        rl = st.selectbox("Current label", core.CLASSES, index=core.CLASSES.index(rule_default), key=f"c_lbl_{tag}",
                          help=GUIDE["risk_label"][2])
    else:
        st.write(f"Rule result: **{rl}**")

    go = st.button("Predict next-quarter label", type="primary", key="c_btn")
    if go:
        b_c = bundle_with_spinner("label", model_name)
    if b_c is None:   # show disclaimer even before the model is run, using cached results if present
        try:
            b_c = get_bundle(DATA_KEY, "label", model_name, DATA) if st.session_state.get("c_trained_" + model_name) else None
        except Exception:
            b_c = None
    if go and b_c is not None:
        st.session_state["c_trained_" + model_name] = True

    with disc:
        st.error("⚠️ **Low accuracy, use with caution.** This label is a rule-based heuristic (not an official or agronomic standard), "
                 "and predicting it one quarter ahead is hard. In our hold-out test the model barely beats, and often does not beat, "
                 "simply repeating the current label. Do not use it for decisions on its own.")
        if b_c is None:
            st.caption("Press **Predict next-quarter label** to train the model and see its hold-out scores here.")
        else:
            m = b_c["metrics"]
            show_table(pd.DataFrame({"Accuracy": [m["acc_model"], m["acc_persistence"], m["acc_majority"], m["chance"]],
                                   "Macro-F1": [m["f1_model"], m["f1_persistence"], m["f1_majority"], np.nan]},
                                  index=[f"{b_c['name']}", "Persistence (next = current label)",
                                         f"Always '{m['majority_label']}' (majority class)", "Random guess (4 classes)"]), nd=3)
            st.caption(f"Hold-out: targets in {m['test_year']} ({m['n_test']:,} rows), trained on earlier years. "
                       "Macro-F1 weighs all four classes equally, so it is the fairer score.")

    if go:
        row = dict(base_row(prov, crop), target_quarter=tq, **pv, growth_rate_yoy=growth, **wx,
                   cur_label_code=core.CLASS_CODE[rl])
        proba = core.predict_label(b_c, row)
        top = int(np.argmax(proba))
        c = st.columns(3)
        c[0].metric(f"Predicted label for {ty} Q{tq}", core.CLASSES[top])
        c[1].metric("Model confidence", f"{proba[top] * 100:.0f}%")
        c[2].metric("Persistence baseline", rl, help="Predicting that the label stays the same as now.")
        bar_chart(core.CLASSES, proba)
        st.caption("Probabilities are the model's own scores and are not calibrated. A flat spread (no class above ~50%) means the model has little real information.")


# ================================================================== FEATURE GUIDE
with tab_g:
    st.subheader("Feature guide: units and descriptions")
    rows = []
    for k, (lab, unit, desc, kind) in GUIDE.items():
        lo, hi = DATA.feature_range(k) if k in DATA.df.columns else (np.nan, np.nan)
        rows.append({"Feature": k, "Name in the app": lab, "Unit": unit, "You enter it?": "yes" if kind == "input" else "auto",
                     "Typical range (5th-95th pct)": "-" if np.isnan(lo) else f"{lo:,.2f} to {hi:,.2f}", "Description": desc})
    show_table(pd.DataFrame(rows).set_index("Feature"))
    st.markdown("""
**Which features does each section use?**

| Section | Features |
|---|---|
| A · Forecast | province, crop, crop group, year, quarter, production lags 1 / 2 / 4 |
| B · Nowcast | everything in A **plus** the 16 weather/ONI variables, ENSO flags, 4-quarter rolling anomalies and climate stress |
| C · Next-quarter label | production (t), lags 1-4, YoY growth, weather of t, rolling anomalies, climate stress, current label |

**Notes**
- Production is **province-level, metric tons per quarter**; small crops can be below 1 t, palay/corn in the tens of thousands.
- The risk label rule: `High` if growth ≥ +10% and stress < P75; `Declining` if growth ≤ −10%; `Risk-prone` if stress ≥ P75 or |ONI| ≥ 1.0; otherwise `Stable`.
- The dataset is a derived, aggregated release (PSA crop tables + NOAA ONI + Open-Meteo-style weather); see the README for limitations.
""")
