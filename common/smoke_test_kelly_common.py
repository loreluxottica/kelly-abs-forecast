# Databricks notebook source
# MAGIC %md
# MAGIC # Smoke test — common/kelly_common.py
# MAGIC Verifica gli helper condivisi su dati sintetici prima del rollout nei 5 notebook.
# MAGIC Da eseguire una volta su Databricks dopo ogni modifica al modulo.

# COMMAND ----------

import os, sys
_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
from common import kelly_common as kc

print("kelly_common importato da:", kc.__file__)
print("QUANTILES:", kc.QUANTILES)
print("STANDARD_COLS:", kc.STANDARD_COLS)

# COMMAND ----------

# ── compute_metrics: NaN-pair-safe ──
a = pd.Series([0.10, 0.20, np.nan, 0.30])
f = pd.Series([0.12, np.nan, 0.25, 0.28])
m = kc.compute_metrics(a, f)
assert m["N"] == 2, m
assert abs(m["MAE"] - 0.02) < 1e-9, m
assert abs(m["Bias"] - 0.0) < 1e-9, m
empty = kc.compute_metrics(pd.Series([np.nan]), pd.Series([np.nan]))
assert empty["N"] == 0 and np.isnan(empty["MAE"])
print("✓ compute_metrics")

# COMMAND ----------

# ── detect_quantile_cols: nomi NP con formattazione variabile ──
fc = pd.DataFrame(columns=["ds", "ID", "yhat1", "yhat1 5.0%", "yhat1 95.0%"])
assert kc.detect_quantile_cols(fc, "yhat1") == ("yhat1 5.0%", "yhat1 95.0%")
fc2 = pd.DataFrame(columns=["ds", "origin-0", "origin-0 5%", "origin-0 95%"])
assert kc.detect_quantile_cols(fc2, "origin-0") == ("origin-0 5%", "origin-0 95%")
try:
    kc.detect_quantile_cols(pd.DataFrame(columns=["ds", "yhat1"]), "yhat1")
    raise AssertionError("doveva sollevare ValueError")
except ValueError:
    pass
print("✓ detect_quantile_cols")

# COMMAND ----------

# ── extract_direct_forecast: clip + quantile crossing ──
fc = pd.DataFrame({
    "ds": pd.to_datetime(["2026-01-05", "2026-01-06"]),
    "ID": ["X", "X"],
    "yhat1":        [0.10, 1.20],   # 1.20 -> clip a 1.0
    "yhat1 5.0%":   [0.15, 0.90],   # 0.15 > point 0.10 -> crossing, forzato a 0.10
    "yhat1 95.0%":  [0.20, 0.95],   # 0.95 < point clipped 1.0 -> forzato a 1.0
})
out = kc.extract_direct_forecast(fc, "Forecast")
assert list(out.columns) == ["ds", "ID", "Forecast", "Forecast_Lower", "Forecast_Upper"]
assert (out["Forecast_Lower"] <= out["Forecast"]).all()
assert (out["Forecast_Upper"] >= out["Forecast"]).all()
assert out["Forecast"].max() <= 1.0
print("✓ extract_direct_forecast")

# COMMAND ----------

# ── mask_bounds_like_point ──
df = pd.DataFrame({
    "Forecast":       [0.1, np.nan],
    "Forecast_Lower": [0.05, 0.10],
    "Forecast_Upper": [0.15, 0.30],
})
df = kc.mask_bounds_like_point(df)
assert df.loc[1, ["Forecast_Lower", "Forecast_Upper"]].isna().all()
assert df.loc[0, "Forecast_Lower"] == 0.05
print("✓ mask_bounds_like_point")

# COMMAND ----------

# ── carry_forward_vintage: freeze lag-1 (point + bounds) ──
prev = pd.DataFrame({
    "ds": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-10"]),
    "ID": ["X"] * 4,
    "Forecast_Vintage": [0.11, np.nan, np.nan, np.nan],
    "Forecast":         [np.nan, 0.22, 0.33, 0.44],
    "Forecast_Lower":   [np.nan, 0.20, 0.30, 0.40],
    "Forecast_Upper":   [np.nan, 0.25, 0.36, 0.48],
})
vintage_all, meta = kc.carry_forward_vintage(prev, pd.Timestamp("2026-06-05"))
# 06-01 vintage storico mantenuto; 06-02/03 congelati (<= freeze); 06-10 futuro -> escluso
assert meta["n_frozen"] == 2
assert len(vintage_all) == 3
assert set(vintage_all["ds"].dt.day) == {1, 2, 3}
assert list(vintage_all.columns) == ["ds", "ID"] + kc.VINTAGE_COLS
_row2 = vintage_all[vintage_all["ds"] == "2026-06-02"].iloc[0]
assert (_row2["Forecast_Vintage"], _row2["Forecast_Vintage_Lower"], _row2["Forecast_Vintage_Upper"]) == (0.22, 0.20, 0.25)

# Tabella precedente con schema vecchio (senza colonne bound) -> bound NaN, nessun crash
prev_old = prev.drop(columns=["Forecast_Lower", "Forecast_Upper"])
v_old, m_old = kc.carry_forward_vintage(prev_old, pd.Timestamp("2026-06-05"))
assert m_old["n_frozen"] == 2
assert v_old["Forecast_Vintage_Lower"].isna().all()

empty_v, empty_meta = kc.carry_forward_vintage(pd.DataFrame(), pd.Timestamp("2026-06-05"))
assert empty_v.empty and empty_meta["n_frozen"] == 0
print("✓ carry_forward_vintage (trio + retrocompatibilita)")

# COMMAND ----------

# ── finalize_output: schema standard, round(4), bound mancanti -> NaN ──
raw = pd.DataFrame({
    "ds": pd.to_datetime(["2026-06-02"]),
    "ID": ["X"],
    "Actual": [0.123456],
    "Forecast_Vintage": [np.nan],
    "Forecast": [0.2],
})
fin = kc.finalize_output(raw)
assert list(fin.columns) == kc.STANDARD_COLS       # 9 colonne (incl. vintage bounds)
assert len(kc.STANDARD_COLS) == 9
assert fin.loc[0, "Actual"] == 0.1235
assert np.isnan(fin.loc[0, "Forecast_Lower"])
assert np.isnan(fin.loc[0, "Forecast_Vintage_Lower"])
print("✓ finalize_output (9 colonne)")

# COMMAND ----------

# ── events helpers ──
ev = {"Super_Bowl": ["2026-02-08"], "Xmas_Eve": ["2025-12-24", "2026-12-24"]}
wide = kc.events_dict_to_wide(ev)
assert set(wide.columns) == {"ds", "Super_Bowl", "Xmas_Eve"}
long = kc.build_future_events_long(ev, "2026-01-01", "2026-03-01")
assert len(long) == 1 and long.iloc[0]["event"] == "Super_Bowl"
assert kc.build_future_events_long(ev, "2027-06-01", "2027-07-01") is None
print("✓ events helpers")

# COMMAND ----------

# ── check_staleness ──
notified = []
try:
    kc.check_staleness(pd.Timestamp.now() - pd.Timedelta(days=30), 14, "test source",
                       notify=lambda t, m: notified.append((t, m)))
    raise AssertionError("doveva sollevare RuntimeError")
except RuntimeError as e:
    assert "DATI NON AGGIORNATI" in str(e)
assert len(notified) == 1
kc.check_staleness(pd.Timestamp.now(), 14, "test source")  # fresco: nessuna eccezione
print("✓ check_staleness")

# COMMAND ----------

# ── read_delta_or_none (solo su Databricks) ──
# Tabella inesistente -> None (primo run); errori diversi -> raise.
assert kc.read_delta_or_none(spark, f"{kc.SCHEMA_QUALIFIED}.tabella_che_non_esiste_xyz") is None
print("✓ read_delta_or_none: tabella mancante -> None")

# Tabella reale -> DataFrame
_t = kc.read_delta_or_none(spark, kc.forecast_table("da"))
print("kelly_da_forecast:", None if _t is None else _t.shape)

# COMMAND ----------

# ── Verifica nomi colonne quantile della versione NP installata ──
# Primo task di smoke su Databricks: un predict minimale e stampa colonne.
from neuralprophet import NeuralProphet

_rng = pd.date_range("2024-01-01", periods=200, freq="D")
_toy = pd.DataFrame({"ds": _rng, "y": (np.sin(np.arange(200) / 7) + 2) / 10})
_m = NeuralProphet(n_forecasts=7, epochs=5, quantiles=kc.QUANTILES,
                   yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
_m.fit(_toy, freq="D")
_fut = _m.make_future_dataframe(_toy, periods=7)
_fc = _m.predict(_fut)
print("Colonne predict (n_lags=0):", _fc.columns.tolist())
_lo, _up = kc.detect_quantile_cols(_fc, "yhat1")
print(f"✓ quantili rilevati: {_lo} / {_up}")

_out = kc.extract_direct_forecast(_fc.assign(ID="TOY"), "Forecast")
assert (_out["Forecast_Lower"] <= _out["Forecast"]).all()
assert (_out["Forecast_Upper"] >= _out["Forecast"]).all()
print(_out.tail(7).to_string(index=False))

# COMMAND ----------

# ── Stesso check per il percorso n_lags>0 (origin-0) ──
_m2 = NeuralProphet(n_lags=14, n_forecasts=7, epochs=5, quantiles=kc.QUANTILES,
                    yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
_toy2 = _toy.assign(ID="TOY")
_m2.fit(_toy2, freq="D")
_fut2 = _m2.make_future_dataframe(_toy2, periods=7)
_fc2 = _m2.predict(_fut2)
print("Colonne predict (n_lags=14):", _fc2.columns.tolist())

_out2 = kc.extract_latest_forecast(_fc2, _m2, "Forecast")
assert list(_out2.columns) == ["ds", "ID", "Forecast", "Forecast_Lower", "Forecast_Upper"]
assert (_out2["Forecast_Lower"] <= _out2["Forecast"]).all()
assert (_out2["Forecast_Upper"] >= _out2["Forecast"]).all()
print(_out2.tail(7).to_string(index=False))
print("\n✓✓ SMOKE TEST COMPLETO — kelly_common pronto per il rollout")
