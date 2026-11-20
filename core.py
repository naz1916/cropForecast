"""
core.py - data preparation, feature definitions and model training for the CropForecast app.
No Streamlit imports here so the logic can be tested on its own.
Feature sets and the next-quarter dataset follow the rolling-origin notebook exactly.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

SEED = 42
CLASSES = ["Stable", "Risk-prone", "High", "Declining"]
CLASS_CODE = {c: i for i, c in enumerate(CLASSES)}

WEATHER = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum",
           "precipitation_hours", "sunshine_duration", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
           "wind_speed_10m_max", "wind_gusts_10m_max", "rain_normal", "temp_normal", "rainfall_deviation_pct",
           "temperature_anomaly_c", "oni_index"]
WEATHER_ALL = WEATHER + ["la_nina", "neutral"]
ROLLING = ["temp_anomaly_rolling", "rain_anomaly_rolling", "climate_stress"]
IDS = ["province_code", "crop_code", "group_code", "qn", "year", "tix"]
LAGS = ["production_lag_1", "production_lag_2", "production_lag_4"]
REG_FORECAST = IDS + LAGS
REG_NOWCAST = IDS + LAGS + WEATHER_ALL + ROLLING
NEXT_FULL = (["province_code", "crop_code", "group_code", "target_quarter", "production", "production_lag_1",
              "production_lag_2", "production_lag_3", "production_lag_4", "growth_rate_yoy"]
             + WEATHER_ALL + ROLLING + ["cur_label_code"])


# --------------------------------------------------------------------------------------------- reading the CSV
def read_dataset(source) -> pd.DataFrame:
    """Read crop_trend_master_2025q4.csv from a path or raw bytes.
    Tolerates a non-UTF-8 encoding and a mangled 'enso_phase_La Niña' column name (the README warns about this)."""
    raw_bytes = source if isinstance(source, (bytes, bytearray)) else None
    last_err = None
    df = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes) if raw_bytes is not None else source, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
    if df is None:
        raise ValueError(f"Could not read the CSV (encoding problem): {last_err}")
    df.columns = [str(c).strip() for c in df.columns]
    # any 'enso_phase_La Ni...' spelling -> canonical name
    fix = {c: "enso_phase_La Niña" for c in df.columns if re.match(r"(?i)^enso_phase_la\s*ni", c) and c != "enso_phase_La Niña"}
    if fix:
        df = df.rename(columns=fix)
    # boolean flags stored as text
    for c in ("enso_phase_La Niña", "enso_phase_Neutral"):
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    return df

# --------------------------------------------------------------------------------------------- data
class Data:
    """Cleaned dataframe + lookup tables."""
    def __init__(self, raw: pd.DataFrame):
        need = ["province", "crop", "crop_group", "year", "quarter", "production", "production_lag_1", "production_lag_2",
                "production_lag_4", "growth_rate_yoy", "risk_label", "enso_phase_La Niña", "enso_phase_Neutral"] + WEATHER + ROLLING
        missing = [c for c in need if c not in raw.columns]
        if missing:
            raise KeyError(f"The CSV is missing required columns: {missing}")
        df = raw.copy()
        df["qn"] = df["quarter"].astype(str).str.extract(r"(\d)")[0].astype(int)
        df["tix"] = df["year"].astype(int) * 4 + df["qn"]
        df = df.drop_duplicates()
        df = df[df.production >= 0].replace([np.inf, -np.inf], np.nan)
        df = df.sort_values(["province", "crop", "tix"]).reset_index(drop=True)
        df["series"] = df.province + "|" + df.crop
        df["la_nina"] = df["enso_phase_La Niña"].astype(int)
        df["neutral"] = df["enso_phase_Neutral"].astype(int)
        self.provinces = sorted(df.province.unique())
        self.crops = sorted(df.crop.unique())
        self.groups = sorted(df.crop_group.unique())
        self.prov_code = {p: i for i, p in enumerate(self.provinces)}
        self.crop_code = {c: i for i, c in enumerate(self.crops)}
        self.group_code = {g: i for i, g in enumerate(self.groups)}
        df["province_code"] = df.province.map(self.prov_code)
        df["crop_code"] = df.crop.map(self.crop_code)
        df["group_code"] = df.crop_group.map(self.group_code)
        self.crop_group_of = df.drop_duplicates("crop").set_index("crop").crop_group.to_dict()
        self.df = df
        self.p75_stress = float(df.climate_stress.quantile(.75))   # threshold used by the README rule
        self.nx = self._build_next(df)
        self.last_year = int(df.year.max())

    @staticmethod
    def _build_next(df: pd.DataFrame) -> pd.DataFrame:
        sk = ["province", "crop", "tix"]
        amb = df.duplicated(sk, keep=False)
        lk = df.loc[~amb, sk + ["year", "production", "risk_label"]]
        nxt = (lk.rename(columns={"production": "next_production", "risk_label": "next_label", "year": "next_year"})
                 .assign(tix=lambda d: d.tix - 1))
        p3 = lk[sk + ["production"]].rename(columns={"production": "production_lag_3"}).assign(tix=lambda d: d.tix + 3)
        nx = df.loc[~amb].merge(nxt, on=sk, how="inner").merge(p3, on=sk, how="inner").reset_index(drop=True)
        nx["cur_label_code"] = nx.risk_label.map(CLASS_CODE)
        nx["y_next"] = nx.next_label.map(CLASS_CODE)
        tt = nx.tix + 1
        nx["target_tix"] = tt
        nx["target_year"] = (tt - 1) // 4
        nx["target_quarter"] = (tt - 1) % 4 + 1
        return nx

    # ---- helpers for the UI -------------------------------------------------------------------
    def crops_in(self, province: str) -> list[str]:
        return sorted(self.df.loc[self.df.province == province, "crop"].unique())

    def series(self, province: str, crop: str) -> pd.Series:
        """production indexed by tix (last value if a key is duplicated)."""
        d = self.df[(self.df.province == province) & (self.df.crop == crop)]
        return d.groupby("tix").production.last()

    def latest_tix(self, province: str, crop: str) -> int:
        s = self.series(province, crop)
        return int(s.index.max()) if len(s) else int(self.df.tix.max())

    def climatology(self, province: str, qn: int) -> dict:
        """Median of every weather / rolling variable for this province and calendar quarter."""
        d = self.df[(self.df.province == province) & (self.df.qn == qn)]
        cols = [c for c in WEATHER + ["temp_anomaly_rolling", "rain_anomaly_rolling"] if c in d.columns]
        return d[cols].median().to_dict()

    def feature_range(self, col: str) -> tuple[float, float]:
        s = self.df[col] if col in self.df.columns else None
        return (float(s.quantile(.05)), float(s.quantile(.95))) if s is not None else (np.nan, np.nan)


def tix_to_yq(tix: int) -> tuple[int, int]:
    return (tix - 1) // 4, (tix - 1) % 4 + 1


def enso_from_oni(oni: float) -> str:
    return "La Niña" if oni <= -0.5 else ("El Niño" if oni >= 0.5 else "Neutral")


def rule_label(growth: float, stress: float, oni: float, p75: float) -> str:
    """README rule used to create `risk_label`."""
    if growth >= .10 and stress < p75: return "High"
    if growth <= -.10: return "Declining"
    if stress >= p75 or abs(oni) >= 1.0: return "Risk-prone"
    return "Stable"


def derive_weather_features(w: dict, rain_roll: float, temp_roll: float) -> dict:
    """Fill the columns that are exact functions of the user inputs (verified 100% on the dataset)."""
    out = dict(w)
    rn = out["rain_normal"]
    out["rainfall_deviation_pct"] = (out["precipitation_sum"] - rn) / rn * 100 if rn > 0 else 0.0
    out["temperature_anomaly_c"] = out["temperature_2m_mean"] - out["temp_normal"]
    ph = enso_from_oni(out["oni_index"])
    out["la_nina"] = int(ph == "La Niña")
    out["neutral"] = int(ph == "Neutral")
    out["temp_anomaly_rolling"] = temp_roll
    out["rain_anomaly_rolling"] = rain_roll
    out["climate_stress"] = abs(rain_roll) + abs(temp_roll)
    return out

# --------------------------------------------------------------------------------------------- models
def available_models() -> list[str]:
    names = ["HistGradientBoosting (sklearn)", "Random Forest"]
    for lib, nm in (("lightgbm", "LightGBM"), ("xgboost", "XGBoost"), ("catboost", "CatBoost")):
        try:
            __import__(lib); names.append(nm)
        except Exception:
            pass
    return names


def make_reg(name: str):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, learning_rate=.05, max_depth=8, subsample=.8, colsample_bytree=.8,
                            tree_method="hist", n_jobs=-1, random_state=SEED)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=300, learning_rate=.05, num_leaves=63, subsample=.8, subsample_freq=1,
                             colsample_bytree=.8, random_state=SEED, n_jobs=-1, verbose=-1)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=300, depth=8, learning_rate=.08, loss_function="RMSE", random_seed=SEED,
                                 verbose=0, thread_count=-1)
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_depth=8, random_state=SEED)


def make_clf(name: str):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, max_features=.7, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=.1, max_depth=6, min_child_weight=3, subsample=.8,
                             colsample_bytree=.8, reg_lambda=3, gamma=.1, objective="multi:softprob",
                             tree_method="hist", n_jobs=-1, random_state=SEED)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=.1, max_depth=7, num_leaves=40, min_child_samples=30,
                              subsample=.8, subsample_freq=1, colsample_bytree=.8, reg_lambda=3, random_state=SEED,
                              n_jobs=-1, verbose=-1)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, depth=6, learning_rate=.1, l2_leaf_reg=3, loss_function="MultiClass",
                                  random_seed=SEED, verbose=0, thread_count=-1)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.1, max_depth=6, random_state=SEED)


def _fit(model, name, X, y, cat_cols):
    if name == "CatBoost":
        X = X.copy()
        for c in cat_cols: X[c] = X[c].astype(int)
        model.fit(X, y, cat_features=[c for c in cat_cols if c in X.columns])
    else:
        model.fit(X, y)
    return model


def _predict(model, name, X, cat_cols, proba=False):
    X = X.copy()
    if name == "CatBoost":
        for c in cat_cols: X[c] = X[c].astype(int)
    return model.predict_proba(X) if proba else np.asarray(model.predict(X)).ravel()


CAT_COLS = ["province_code", "crop_code", "group_code"]


def train_regressor(data: Data, name: str, feats: list[str]) -> dict:
    """Holdout on the latest year (train = earlier years), then refit on everything."""
    df, Y = data.df, data.last_year
    tr, te = df.year < Y, df.year == Y
    m = _fit(make_reg(name), name, df.loc[tr, feats], df.loc[tr, "production"], CAT_COLS)
    p = np.clip(_predict(m, name, df.loc[te, feats], CAT_COLS), 0, None)
    y = df.loc[te, "production"].values
    mae = float(mean_absolute_error(y, p))
    metrics = dict(test_year=Y, n_test=int(te.sum()), MAE_model=mae,
                   WAPE_model=float(100 * np.abs(y - p).sum() / np.abs(y).sum()),
                   MAE_naive=float(mean_absolute_error(y, df.loc[te, "production_lag_1"])),
                   MAE_seasonal_naive=float(mean_absolute_error(y, df.loc[te, "production_lag_4"])))
    final = _fit(make_reg(name), name, df[feats], df["production"], CAT_COLS)
    return dict(model=final, feats=feats, name=name, metrics=metrics)


def train_classifier(data: Data, name: str) -> dict:
    nx, Y = data.nx, int(data.nx.target_year.max())
    tr, te = nx.target_year < Y, nx.target_year == Y
    m = _fit(make_clf(name), name, nx.loc[tr, NEXT_FULL], nx.loc[tr, "y_next"].values, CAT_COLS)
    p = _predict(m, name, nx.loc[te, NEXT_FULL], CAT_COLS).astype(int)
    y = nx.loc[te, "y_next"].values
    maj = np.bincount(nx.loc[tr, "y_next"], minlength=4).argmax()
    pers = nx.loc[te, "cur_label_code"].values
    metrics = dict(test_year=Y, n_test=int(te.sum()),
                   acc_model=float(accuracy_score(y, p)), f1_model=float(f1_score(y, p, average="macro")),
                   acc_majority=float(accuracy_score(y, np.full_like(y, maj))),
                   f1_majority=float(f1_score(y, np.full_like(y, maj), average="macro")),
                   acc_persistence=float(accuracy_score(y, pers)),
                   f1_persistence=float(f1_score(y, pers, average="macro")),
                   majority_label=CLASSES[int(maj)],
                   chance=0.25)
    final = _fit(make_clf(name), name, nx[NEXT_FULL], nx["y_next"].values, CAT_COLS)
    return dict(model=final, feats=NEXT_FULL, name=name, metrics=metrics)


def predict_production(bundle: dict, row: dict) -> float:
    X = pd.DataFrame([row])[bundle["feats"]]
    return float(max(_predict(bundle["model"], bundle["name"], X, CAT_COLS)[0], 0.0))


def predict_label(bundle: dict, row: dict) -> np.ndarray:
    X = pd.DataFrame([row])[bundle["feats"]]
    proba = _predict(bundle["model"], bundle["name"], X, CAT_COLS, proba=True)[0]
    out = np.zeros(4)
    for cls, pr in zip(bundle["model"].classes_, proba): out[int(cls)] = pr
    return out
