# =============================================================================
# kelly_common.py — helper condivisi per i notebook Kelly Absenteeism Forecast
# (ATL, COL, DA, MX, IT). Brasile escluso.
#
# Import dai notebook (dopo dbutils.library.restartPython()):
#     import os, sys
#     _REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
#     if _REPO_ROOT not in sys.path:
#         sys.path.insert(0, _REPO_ROOT)
#     from common import kelly_common as kc
# =============================================================================
import logging
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd

# ── Costanti standard ────────────────────────────────────────────────────────
# Intervallo di previsione 90% (quantile regression NeuralProphet)
QUANTILES = [0.05, 0.95]

# Schema standard delle Delta table di output (tutte le geografie).
# I bound vintage sono APPESI in coda per non spostare le colonne esistenti in BI.
STANDARD_COLS = [
    "ds", "ID", "Actual", "Forecast_Vintage",
    "Forecast", "Forecast_Lower", "Forecast_Upper",
    "Forecast_Vintage_Lower", "Forecast_Vintage_Upper",
]
NUMERIC_COLS = STANDARD_COLS[2:]
VINTAGE_COLS = ["Forecast_Vintage", "Forecast_Vintage_Lower", "Forecast_Vintage_Upper"]
FORECAST_COLS = ["Forecast", "Forecast_Lower", "Forecast_Upper"] + VINTAGE_COLS

ROUND_DECIMALS = 4


# ── Logging / timing ─────────────────────────────────────────────────────────
def get_logger(name: str = "kelly") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger(name)


@contextmanager
def timed(label: str, log: logging.Logger | None = None, timings: dict | None = None):
    _log = log or logging.getLogger("kelly")
    _log.info(f"▶ {label}...")
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if timings is not None:
        timings[label] = round(elapsed, 2)
    _log.info(f"✓ {label} — {elapsed:.1f}s")


# ── Metriche ─────────────────────────────────────────────────────────────────
def compute_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    """MAE / Bias / RMSE / SMAPE / WMAE (peso 2x sottostima) / N.
    Le coppie con NaN in actual o forecast sono escluse."""
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    f = pd.to_numeric(forecast, errors="coerce").to_numpy(dtype=float)
    ok = ~(np.isnan(a) | np.isnan(f))
    a, f = a[ok], f[ok]
    if len(a) == 0:
        return {"MAE": np.nan, "Bias": np.nan, "RMSE": np.nan,
                "SMAPE": np.nan, "WMAE": np.nan, "N": 0}
    err = f - a
    num = 2 * np.abs(err)
    den = np.abs(a) + np.abs(f)
    w = np.where(f < a, 2, 1)  # peso 2x per sottostima (under)
    return {
        "MAE":   round(float(np.mean(np.abs(err))), 4),
        "Bias":  round(float(np.mean(err)), 4),
        "RMSE":  round(float(np.sqrt(np.mean(err ** 2))), 4),
        "SMAPE": round(float(np.mean(num[den != 0] / den[den != 0]) * 100), 2),
        "WMAE":  round(float((w * np.abs(err)).sum() / w.sum()), 4),
        "N":     int(len(a)),
    }


# ── Eventi custom ────────────────────────────────────────────────────────────
def events_dict_to_wide(events: dict) -> pd.DataFrame:
    """{'nome': [date]} -> DataFrame wide: ds + una colonna 0/1 per evento."""
    df_events = pd.concat(
        [pd.DataFrame({"event": name, "ds": pd.to_datetime(dates)})
         for name, dates in events.items()],
        ignore_index=True,
    )
    return (
        df_events
        .assign(value=1)
        .pivot_table(index="ds", columns="event", values="value", aggfunc="max")
        .fillna(0)
        .reset_index()
    )


def build_future_events_long(events: dict, start_ds, end_ds) -> pd.DataFrame | None:
    """DataFrame long {'event','ds'} per le occorrenze nel range (start_ds, end_ds].
    None se nessuna occorrenza (make_future_dataframe accetta events_df=None)."""
    start_ds, end_ds = pd.Timestamp(start_ds), pd.Timestamp(end_ds)
    records = []
    for name, dates in events.items():
        for d in pd.to_datetime(dates):
            if start_ds < d <= end_ds:
                records.append({"event": name, "ds": d})
    return pd.DataFrame(records) if records else None


# ── Estrazione quantili dal predict NeuralProphet ────────────────────────────
def detect_quantile_cols(fcst: pd.DataFrame, base_col: str,
                         quantiles: list = QUANTILES) -> tuple:
    """Trova le colonne quantile lower/upper per base_col ('yhat1' o 'origin-0').
    NP 0.9.x le nomina f'{base_col} {q*100}%' (es. 'yhat1 5.0%'), ma il formato
    del float varia tra versioni -> regex, mai hardcodare i nomi."""
    pattern = re.compile(rf"^{re.escape(base_col)}\s+(\d+(?:\.\d+)?)\s*%$")
    found = {}
    for c in fcst.columns:
        m = pattern.match(str(c))
        if m:
            found[float(m.group(1))] = c
    if len(found) < 2:
        raise ValueError(
            f"Colonne quantile per '{base_col}' non trovate "
            f"(attese ~{[q * 100 for q in quantiles]}%). "
            f"Colonne disponibili: {fcst.columns.tolist()}"
        )
    lower_col = found[min(found)]
    upper_col = found[max(found)]
    return lower_col, upper_col


def _enforce_bounds(df: pd.DataFrame, point: str, lower: str, upper: str) -> pd.DataFrame:
    """Clip [0,1] e garanzia Lower <= point <= Upper (quantile crossing)."""
    for c in (point, lower, upper):
        df[c] = df[c].clip(0, 1)
    df[lower] = np.minimum(df[lower], df[point])
    df[upper] = np.maximum(df[upper], df[point])
    return df


def extract_direct_forecast(forecast_df: pd.DataFrame, col_name: str,
                            with_bounds: bool = True) -> pd.DataFrame:
    """Percorso n_lags=0 (COL, DA): yhat1 e' la prediction diretta per riga.
    Ritorna ds, ID, {col}, [{col}_Lower, {col}_Upper]."""
    if "yhat1" not in forecast_df.columns:
        raise ValueError(f"yhat1 non trovato. Colonne: {forecast_df.columns.tolist()}")
    if not with_bounds:
        result = forecast_df[["ds", "ID", "yhat1"]].copy()
        result["yhat1"] = result["yhat1"].clip(0, 1)
        return result.rename(columns={"yhat1": col_name})
    lo, up = detect_quantile_cols(forecast_df, "yhat1")
    result = forecast_df[["ds", "ID", "yhat1", lo, up]].copy()
    result = _enforce_bounds(result, "yhat1", lo, up)
    return result.rename(columns={
        "yhat1": col_name, lo: f"{col_name}_Lower", up: f"{col_name}_Upper",
    })


def _bounds_from_diagonal(subset: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    """Fallback se get_latest_forecast non propaga le colonne quantile:
    ricostruisce la diagonale dal predict completo. Le righe di latest sono le
    ultime len(latest) righe di subset; la riga i-esima usa 'yhat{i+1} q%'."""
    tail = subset.tail(len(latest)).reset_index(drop=True)
    lowers, uppers = [], []
    for i in range(len(tail)):
        lo, up = detect_quantile_cols(subset, f"yhat{i + 1}")
        lowers.append(tail.at[i, lo])
        uppers.append(tail.at[i, up])
    out = latest.reset_index(drop=True).copy()
    out["_lower"] = lowers
    out["_upper"] = uppers
    return out


def extract_latest_forecast(forecast_df: pd.DataFrame, model_for_id,
                            col_name: str, with_bounds: bool = True) -> pd.DataFrame:
    """Percorso n_lags>0 (ATL, MX, IT): per ogni ID, model.get_latest_forecast
    -> colonna 'origin-0' (+ quantili). model_for_id: modello NeuralProphet
    oppure callable(uid) -> modello (ATL usa modelli diversi per gruppo).
    Ritorna ds, ID, {col}, [{col}_Lower, {col}_Upper]."""
    get_model = model_for_id if callable(model_for_id) else (lambda uid: model_for_id)
    parts = []
    for uid in forecast_df["ID"].unique():
        subset = forecast_df[forecast_df["ID"] == uid]
        latest = get_model(uid).get_latest_forecast(subset).copy()
        latest["ID"] = uid
        if with_bounds:
            try:
                lo, up = detect_quantile_cols(latest, "origin-0")
                latest = latest.rename(columns={lo: "_lower", up: "_upper"})
            except ValueError:
                # get_latest_forecast ha scartato i quantili -> diagonale manuale
                latest = _bounds_from_diagonal(subset, latest)
        parts.append(latest)

    out = pd.concat(parts, ignore_index=True)
    keep = ["ds", "ID", "origin-0"] + (["_lower", "_upper"] if with_bounds else [])
    out = out[keep].copy()
    if with_bounds:
        out = _enforce_bounds(out, "origin-0", "_lower", "_upper")
        return out.rename(columns={
            "origin-0": col_name,
            "_lower": f"{col_name}_Lower", "_upper": f"{col_name}_Upper",
        })
    out["origin-0"] = out["origin-0"].clip(0, 1)
    return out.rename(columns={"origin-0": col_name})


def mask_bounds_like_point(df: pd.DataFrame,
                           point_col: str = "Forecast",
                           lower_col: str = "Forecast_Lower",
                           upper_col: str = "Forecast_Upper") -> pd.DataFrame:
    """Dove il point forecast e' NaN (off-day / soglia / non lavorativo),
    anche i bound devono essere NaN. Chiamare DOPO tutti i mascheramenti."""
    mask = df[point_col].isna()
    df.loc[mask, [lower_col, upper_col]] = np.nan
    return df


# ── Delta I/O ────────────────────────────────────────────────────────────────
_TABLE_NOT_FOUND_RE = re.compile(
    r"TABLE_OR_VIEW_NOT_FOUND|not found|cannot be found|doesn't exist|does not exist",
    re.IGNORECASE,
)


def read_delta_or_none(spark, table_name: str, columns: list | None = None):
    """Legge una Delta table in pandas. Ritorna None SOLO se la tabella non
    esiste (primo run); ogni altro errore (permessi, cluster, schema) viene
    rilanciato — un except generico azzererebbe silenziosamente lo storico
    Forecast_Vintage."""
    try:
        from pyspark.errors import AnalysisException  # DBR recenti
    except ImportError:
        from pyspark.sql.utils import AnalysisException
    try:
        sdf = spark.table(table_name)
        if columns:
            sdf = sdf.select(*columns)
        pdf = sdf.toPandas()
    except AnalysisException as e:
        if _TABLE_NOT_FOUND_RE.search(str(e)):
            return None
        raise
    if "ds" in pdf.columns:
        pdf["ds"] = pd.to_datetime(pdf["ds"])
    return pdf


def carry_forward_vintage(prev_df: pd.DataFrame,
                          freeze_until: pd.Timestamp) -> tuple:
    """Logica lag-1: congela Forecast (+ Forecast_Lower/Upper) del run precedente
    per le date ormai trascorse (<= freeze_until) nel trio Forecast_Vintage
    (+ _Lower/_Upper), mantenendo il vintage gia' accumulato. Tollera tabelle
    precedenti senza le colonne bound (schema pre-quantili) -> NaN.
    Ritorna (vintage_all[ds, ID, VINTAGE_COLS], meta dict)."""
    if prev_df is None or prev_df.empty:
        # dtypes espliciti: un ds object su frame vuoto rompe il merge con datetime
        empty = pd.DataFrame({
            "ds": pd.Series(dtype="datetime64[ns]"),
            "ID": pd.Series(dtype="object"),
            **{c: pd.Series(dtype="float64") for c in VINTAGE_COLS},
        })
        return empty, {"last_vintage_date": pd.NaT, "n_frozen": 0}

    prev_df = prev_df.copy()
    prev_df["ds"] = pd.to_datetime(prev_df["ds"])
    # Colonne mancanti (tabella con schema vecchio) -> NaN
    for c in ["Forecast_Lower", "Forecast_Upper"] + VINTAGE_COLS:
        if c not in prev_df.columns:
            prev_df[c] = np.nan

    last_vintage_date = prev_df.loc[prev_df["Forecast_Vintage"].notna(), "ds"].max()
    if pd.isna(last_vintage_date):
        last_vintage_date = prev_df["ds"].min() - pd.Timedelta(days=1)

    # Il freeze e' guidato dal point forecast: i bound seguono la stessa maschera
    mask_freeze = (
        (prev_df["ds"] > last_vintage_date)
        & (prev_df["ds"] <= freeze_until)
        & prev_df["Forecast"].notna()
    )
    frozen = (
        prev_df.loc[mask_freeze, ["ds", "ID", "Forecast", "Forecast_Lower", "Forecast_Upper"]]
        .rename(columns={
            "Forecast": "Forecast_Vintage",
            "Forecast_Lower": "Forecast_Vintage_Lower",
            "Forecast_Upper": "Forecast_Vintage_Upper",
        })
    )
    vintage_all = (
        pd.concat(
            [prev_df.loc[prev_df["Forecast_Vintage"].notna(),
                         ["ds", "ID"] + VINTAGE_COLS],
             frozen],
            ignore_index=True,
        )
        .drop_duplicates(["ds", "ID"], keep="last")
    )
    meta = {"last_vintage_date": last_vintage_date, "n_frozen": int(mask_freeze.sum())}
    return vintage_all, meta


def finalize_output(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Schema standard: colonne STANDARD_COLS, numeriche float64, round(4).
    Le colonne bound mancanti vengono aggiunte come NaN."""
    out = merged_df.copy()
    for c in NUMERIC_COLS:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64").round(ROUND_DECIMALS)
    out["ds"] = pd.to_datetime(out["ds"])
    return out[STANDARD_COLS].sort_values(["ds", "ID"]).reset_index(drop=True)


def write_forecast_table(spark, merged_df: pd.DataFrame, table_name: str,
                         drop_empty_rows: bool = True) -> int:
    """Scrive lo schema standard 7 colonne su Delta (overwrite + overwriteSchema).
    drop_empty_rows: scarta le righe con tutte le colonne numeriche NaN."""
    export_df = finalize_output(merged_df)
    if drop_empty_rows:
        export_df = export_df[export_df[NUMERIC_COLS].notna().any(axis=1)]
    spark_df = spark.createDataFrame(export_df)
    (
        spark_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )
    return len(export_df)


# ── Notifiche / validazione ──────────────────────────────────────────────────
def notify_teams(webhook_url: str, title: str, message: str,
                 job: str, notebook: str, log: logging.Logger | None = None) -> None:
    """POST adaptive card sul webhook Teams (Power Automate). Non solleva mai."""
    import requests as _req
    _log = log or logging.getLogger("kelly")
    card = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {"type": "TextBlock", "text": f"⚠️ {title}", "weight": "Bolder",
                     "size": "Medium", "color": "Attention"},
                    {"type": "TextBlock", "text": message, "wrap": True},
                    {"type": "FactSet", "facts": [
                        {"title": "Job", "value": job},
                        {"title": "Notebook", "value": notebook},
                        {"title": "Timestamp",
                         "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                    ]},
                ],
            }
        }]
    }
    try:
        resp = _req.post(webhook_url, json=card, timeout=10)
        if resp.status_code in (200, 202):
            _log.info("✓ Notifica Teams inviata con successo")
        else:
            _log.warning(f"Teams webhook risposta: {resp.status_code} — {resp.text}")
    except Exception as e:
        _log.warning(f"Impossibile inviare notifica Teams: {e}")


def check_staleness(max_raw_date, max_days: int, source_desc: str,
                    notify=None, log: logging.Logger | None = None) -> int:
    """Solleva RuntimeError (dopo notify) se l'ultimo dato e' piu' vecchio di
    max_days. notify: callable(title, message) opzionale. Ritorna days_stale."""
    _log = log or logging.getLogger("kelly")
    max_raw_date = pd.Timestamp(max_raw_date)
    days_stale = (datetime.now() - max_raw_date.to_pydatetime()).days
    if days_stale > max_days:
        msg = (f"DATI NON AGGIORNATI — ultima data disponibile: "
               f"{max_raw_date.date()} ({days_stale} giorni fa). "
               f"Aggiornare {source_desc}.")
        if notify is not None:
            notify("Dati non aggiornati", msg)
        raise RuntimeError(f"⚠️ {msg}")
    _log.info(f"✅ Freschezza dati OK — ultima data: {max_raw_date.date()} "
              f"({days_stale} giorni fa)")
    return days_stale
