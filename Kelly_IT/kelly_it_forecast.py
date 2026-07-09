# Databricks notebook source
# DBTITLE 1,Kelly IT — Weekly Forecast Pipeline
# MAGIC %md
# MAGIC # Kelly IT — Weekly Absenteeism Forecast Pipeline
# MAGIC
# MAGIC **Version:** 3.1 — Weekly Automated Forecast
# MAGIC **Data Source:** Excel files (OneDrive / Volume)
# MAGIC **Granularità:** giornaliera per reparto → forecast giornaliero
# MAGIC **Model:** NeuralProphet (`n_lags=21`, `n_forecasts=365`, `yearly_seasonality=20`, `weekly_seasonality=5`, `quantiles=[0.05, 0.95]`)
# MAGIC **Vintage:** Lag-1 accumulato (finestra mobile 4 settimane)
# MAGIC **Country holidays:** IT (±1 giorno)
# MAGIC **Custom events:** Apertura Scuole, Chiusura Scuole, Ramadan, Extra Festività
# MAGIC
# MAGIC ---
# MAGIC ### Changelog v3.1 — 2026-07-08 (schema standard + intervalli di previsione)
# MAGIC - **Prediction interval 90%**: `quantiles=[0.05, 0.95]`; nuove colonne `Forecast_Lower`/`Forecast_Upper`
# MAGIC   nella Delta table (schema standard 7 colonne per tutte le geografie).
# MAGIC - **Fix `fillna(1)` (bug)**: i giorni senza dati NON vengono piu riempiti con y=1 (100% assenza fittizia
# MAGIC   che entrava nel training). I giorni non osservati restano NaN; il training scarta le righe NaN
# MAGIC   (approccio MX/ATL). ⚠️ In Power BI i giorni chiusi ora sono VUOTI, non piu 100%.
# MAGIC - **Convenzione output allineata alle altre geografie**: weekend e forecast >0.75 → NaN (prima: 1.0).
# MAGIC - **Fix `m.fit` senza `freq='D'`** (prima si affidava all'auto-inferenza).
# MAGIC - **Fix eventi futuri inerti (bug)**: `make_future_dataframe` ora riceve `events_df` — prima gli eventi
# MAGIC   custom valevano 0 su tutto l'orizzonte di forecast. Estese anche le date eventi al 2026/2027
# MAGIC   (l'orizzonte è 365 giorni).
# MAGIC - **Fix vintage read**: `except Exception` sostituito da gestione esplicita table-not-found.
# MAGIC - **Modulo condiviso `common/kelly_common.py`**.
# MAGIC
# MAGIC *Convertito da `kelly_weekly.py` (script Windows Task Scheduler)*

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports
import os
import gc
import time
import shutil
import logging
import tempfile
from contextlib import contextmanager
from datetime import timedelta, datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from neuralprophet import NeuralProphet, save as np_save, set_random_seed
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Modulo condiviso (repo root)
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common import kelly_common as kc

print('Libraries loaded.')

# COMMAND ----------

# DBTITLE 1,Configuration & Paths
# =============================================================================
# PERCORSI — adattati per Databricks (Unity Catalog Volumes)
# Su Windows usava OneDrive; qui si usa il volume UC.
# =============================================================================
VOLUME_BASE = "/Volumes/sbx-logistics/kelly/kelly_it_volume"

BASE_PATH        = Path(f"{VOLUME_BASE}/input")
OUTPUT_PATH      = Path(f"{VOLUME_BASE}/output")
PLOT_PATH        = Path(f"{VOLUME_BASE}/plots")
LOG_DIR          = Path(f"{VOLUME_BASE}/logs")
HISTORY_CSV      = Path(f"{VOLUME_BASE}/reports/kelly_run_history.csv")
LATEST_MODEL_PKL = OUTPUT_PATH / "kelly_model_latest.pkl"

MODEL_VERSION = "v3.1"
VINTAGE_WEEKS = 4  # Finestra mobile del Forecast_Vintage accumulato

for p in [OUTPUT_PATH, PLOT_PATH, LOG_DIR, HISTORY_CSV.parent, BASE_PATH]:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        print(f"⚠ Impossibile creare {p} — il volume potrebbe non essere montato.")

COLS_TO_DROP = ["Voce assenteismo", "Tipo contratto", "Assenteismo"]

print('Configuration OK.')

# COMMAND ----------

# DBTITLE 1,Logging & Timing
# =============================================================================
# LOGGING & TIMING
# =============================================================================
RUN_TS = datetime.now()
RUN_ID = RUN_TS.strftime('%Y%m%d_%H%M%S')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

_timings: dict[str, float] = {}

@contextmanager
def timed(label: str):
    log.info(f">> {label}...")
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    _timings[label] = round(elapsed, 2)
    log.info(f"[OK] {label} - {elapsed:.1f}s")

script_start = time.perf_counter()
print(f'RUN_ID: {RUN_ID}')

# COMMAND ----------

# DBTITLE 1,1. Data Loading
# =============================================================================
# 1. CARICAMENTO DATI
# =============================================================================
with timed("Caricamento dati"):
    def load_file(filename: str, skiprows: int = 0) -> pd.DataFrame:
        return pd.read_excel(
            BASE_PATH / filename, skiprows=skiprows,
            usecols=lambda c: c not in COLS_TO_DROP
        )

    files_standard = [
        "Assenteismo Logistica 2023.xlsx",
        "Assenteismo Logistica 2024.xlsx",
        "Assenteismo Logistica 2025.xlsx",
    ]
    dfs = [load_file(f) for f in files_standard]

    df4 = load_file("Assenteismo Logistica.xlsx", skiprows=2)
    df4.dropna(subset=["Giorno"], inplace=True)
    df4["Reparto"] = df4["Reparto"].ffill()
    dfs.append(df4)

    df_grouped = (
        pd.concat(dfs, ignore_index=True)
        .groupby(["Reparto", "Giorno"], as_index=False)[["Ore Assenteismo", "Ore Teoriche"]]
        .sum()
    )
    df_grouped["Assenteismo"] = df_grouped["Ore Assenteismo"] / df_grouped["Ore Teoriche"]

n_raw_rows = len(pd.concat(dfs, ignore_index=True))
log.info(f"Righe raw totali: {n_raw_rows:,}")
log.info(f"Ultima osservazione nei dati: {df_grouped['Giorno'].max()}")

# COMMAND ----------

# DBTITLE 1,2. Complete DataFrame
# =============================================================================
# 2. CREAZIONE DATAFRAME COMPLETO
# =============================================================================
with timed("Creazione dataframe completo"):
    df_general = (
        df_grouped
        .groupby("Giorno")[["Ore Assenteismo", "Ore Teoriche"]]
        .sum()
        .reset_index()
        .assign(Assenteismo=lambda x: x["Ore Assenteismo"] / x["Ore Teoriche"], ID="General")
        .drop(columns=["Ore Assenteismo", "Ore Teoriche"])
        .rename(columns={"Giorno": "ds", "Assenteismo": "y"})
    )

    df_grouped = (
        df_grouped
        .drop(columns=["Ore Assenteismo", "Ore Teoriche"])
        .rename(columns={"Giorno": "ds", "Assenteismo": "y", "Reparto": "ID"})
        .pipe(lambda df: pd.concat([df, df_general], ignore_index=True))
        .assign(ds=lambda x: pd.to_datetime(x["ds"]))
    )

    min_date, max_date = df_grouped["ds"].min(), df_grouped["ds"].max()

    full_index = pd.MultiIndex.from_product(
        [pd.date_range(start=min_date, end=max_date, freq="D"), df_grouped["ID"].unique()],
        names=["ds", "ID"],
    )
    # v3.1 FIX: niente fillna(1) — i giorni senza dati restano NaN.
    # Prima venivano riempiti con y=1 (100% assenza fittizia) che entrava
    # direttamente nel training (IT non filtra i giorni per DOW).
    merged = (
        pd.DataFrame(index=full_index)
        .reset_index()
        .merge(df_grouped, on=["ds", "ID"], how="left")
    )

log.info(f"Dataset completo: {merged.shape} | {min_date.date()} → {max_date.date()}")

# COMMAND ----------

# DBTITLE 1,3. ID Filtering
# =============================================================================
# 3. FILTRAGGIO ID
# =============================================================================
with timed("Filtraggio ID"):
    # v3.1: senza fillna(1) il criterio "y medio > 0.6" non intercetta piu i
    # reparti dormienti (media calcolata solo sui giorni osservati). Si esclude
    # quindi anche chi non ha NESSUNA osservazione negli ultimi 100 giorni.
    _recent = merged[merged["ds"].between(max_date - pd.Timedelta(days=100), max_date)]
    _stats  = _recent.groupby("ID")["y"].agg(["mean", "count"])
    ids_to_exclude = _stats[(_stats["mean"] > 0.6) | (_stats["count"] == 0)].index
    df = merged[~merged["ID"].isin(ids_to_exclude)]
    all_IDs = df["ID"].unique()

log.info(f"ID esclusi (y medio > 0.6 o nessun dato recente): {list(ids_to_exclude)}")
log.info(f"ID da forecastare ({len(all_IDs)}): {all_IDs.tolist()}")

# COMMAND ----------

# DBTITLE 1,4. Events
# =============================================================================
# 4. EVENTI
# =============================================================================
with timed("Preparazione eventi"):
    # v3.1: date estese al 2026/2027 — l'orizzonte di forecast e' 365 giorni,
    # senza occorrenze future gli eventi sono inerti sull'intera previsione.
    # Le date scolastiche 2026/2027 sono STIME sul calendario Veneto: da
    # confermare quando la Regione pubblica il calendario ufficiale.
    important_events = {
        "Apertura Scuole": [
            "2023-09-13", "2023-09-14", "2023-09-15",
            "2024-09-11", "2024-09-12", "2024-09-13",
            "2025-09-10", "2025-09-11", "2025-09-12",
            "2026-09-14", "2026-09-15", "2026-09-16",   # stima
            "2027-09-13", "2027-09-14", "2027-09-15",   # stima
        ],
        "Chiusura Scuole": [
            "2023-06-06", "2023-06-07", "2023-06-08",
            "2024-06-06", "2024-06-07", "2024-06-08",
            "2025-06-05", "2025-06-06", "2025-06-07",
            "2026-06-04", "2026-06-05", "2026-06-06",   # stima
            "2027-06-03", "2027-06-04", "2027-06-05",   # stima
        ],
        "Ramadan": [
            "2023-03-23", "2023-04-21",
            "2024-03-11", "2024-04-09",
            "2025-03-01", "2025-03-30",
            "2026-02-18", "2026-03-19",
            "2027-02-08", "2027-03-09",                  # stima lunare
        ],
        "Extra Festività": [
            "2023-06-29", "2024-06-29", "2025-06-29", "2026-06-29", "2027-06-29",
            "2023-12-08", "2024-12-08", "2025-12-08", "2026-12-08", "2027-12-08",
            "2023-12-24", "2024-12-24", "2025-12-24", "2026-12-24", "2027-12-24",
            "2023-12-31", "2024-12-31", "2025-12-31", "2026-12-31", "2027-12-31",
        ],
    }

    # v3.1: pivot wide condiviso (robusto a eventi sovrapposti sulla stessa data,
    # il vecchio merge+get_dummies duplicava le righe in quel caso)
    df_events_wide = kc.events_dict_to_wide(important_events)
    event_cols = list(important_events.keys())

    df = df.merge(df_events_wide, on="ds", how="left")
    df[event_cols] = df[event_cols].fillna(0).astype(int)
log.info(f"Eventi applicati: {event_cols}")

# COMMAND ----------

# DBTITLE 1,5. Train Split
# =============================================================================
# 5. SPLIT TRAIN
# =============================================================================
start_date = pd.Timestamp("2022-01-01")
split_date = max_date - timedelta(days=2)

# v3.1: righe y=NaN rimosse PRIMA del fit (approccio MX/ATL) — i giorni non
# osservati non entrano nel training e si evita il bug metrics-NaN di
# NeuralProphet 0.9.0 con drop_missing sui global models.
train_df = df[df["ds"].between(start_date, split_date)]
_n_nan_removed = train_df["y"].isna().sum()
train_df = train_df[train_df["y"].notna()].reset_index(drop=True)

log.info(f"Train: {train_df['ds'].min().date()} — {train_df['ds'].max().date()} ({len(train_df)} righe, {_n_nan_removed} righe NaN rimosse)")

# COMMAND ----------

# DBTITLE 1,6. Model Training
# =============================================================================
# 6. TRAINING MODELLO
# =============================================================================
import torch, functools

# PyTorch 2.6+: override torch.load per weights_only=False
_orig = torch.serialization.load

@functools.wraps(_orig)
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig(*args, **kwargs)

torch.load = _patched_load

M_CONFIG = dict(
    n_lags=21,
    n_forecasts=365,
    n_changepoints=3,
    trend_reg=0.5,
    seasonality_reg=0.5,
    yearly_seasonality=20,
    weekly_seasonality=5,
    trend_global_local="local",
    season_global_local="local",
    seasonality_mode="additive",
    # v3.1: intervallo di previsione 90% (pinball loss aggiuntiva sui quantili)
    quantiles=kc.QUANTILES,
)

with timed("Training modello"):
    m = NeuralProphet(**M_CONFIG)
    m = m.add_country_holidays("IT", lower_window=-1, upper_window=1)
    m.add_events(["Chiusura Scuole", "Apertura Scuole", "Ramadan", "Extra Festività"])
    m.set_plotting_backend("plotly")
    # v3.1 FIX: freq='D' esplicito (prima auto-inferenza)
    metrics = m.fit(train_df, freq="D", learning_rate=0.01)

final_loss = float(metrics["Loss"].iloc[-1]) if metrics is not None and "Loss" in metrics.columns else None
n_epochs   = len(metrics) if metrics is not None else None
log.info(f"Loss finale: {final_loss}  |  Epoche: {n_epochs}")

# COMMAND ----------

# DBTITLE 1,7. Save Checkpoint
# =============================================================================
# 7. SALVATAGGIO CHECKPOINT LATEST
# =============================================================================
np_save(m, str(LATEST_MODEL_PKL))
log.info(f"Checkpoint latest salvato: {LATEST_MODEL_PKL}")

# COMMAND ----------

# DBTITLE 1,8. Forecast
# =============================================================================
# 8. FORECAST
# =============================================================================
with timed("Generazione forecast"):
    # v3.1 FIX: events_df con le occorrenze future — senza, gli eventi custom
    # valgono 0 su tutto l'orizzonte (bug: Ramadan/scuole/festivita inerti).
    _train_max = train_df["ds"].max()
    future_events_long = kc.build_future_events_long(
        important_events, _train_max, _train_max + pd.Timedelta(days=365)
    )
    _n_fut_ev = len(future_events_long) if future_events_long is not None else 0
    log.info(f"Eventi futuri nell'orizzonte 365gg: {_n_fut_ev} occorrenze")

    future   = m.make_future_dataframe(train_df, periods=365, events_df=future_events_long)
    forecast = m.predict(future)

log.info(f"Forecast generato: {len(forecast)} righe")

# COMMAND ----------

# DBTITLE 1,9. Post-processing + Lag-1 Vintage
# =============================================================================
# 9. POST-PROCESSING FORECAST + LAG-1 VINTAGE
# =============================================================================
with timed("Post-processing forecast + lag-1 vintage"):

    # v3.1: helper condiviso (origin-0 + quantili 5%/95%) + convenzione output
    # allineata alle altre geografie: weekend e valori >0.75 → NaN (prima: 1.0).
    # ⚠️ Power BI: i giorni chiusi ora sono vuoti, non piu barre al 100%.
    df_forecast = kc.extract_latest_forecast(forecast, m, col_name="Forecast")
    _is_weekend = df_forecast["ds"].dt.dayofweek.isin([5, 6])
    df_forecast.loc[df_forecast["Forecast"] > 0.75, "Forecast"] = np.nan
    df_forecast.loc[_is_weekend, "Forecast"] = np.nan
    df_forecast = kc.mask_bounds_like_point(df_forecast)

    # --- Lag-1 Vintage: lettura da Delta table (con fallback a seed Excel) ---
    DELTA_TABLE = "`sbx-logistics`.kelly.kelly_it_forecast"
    SEED_FILE = OUTPUT_PATH / "Kelly_v25_DB_seed.xlsx"

    def _load_previous_forecast_vintage(current_max_date) -> pd.Series | None:
        """
        Costruisce il Forecast_Vintage ACCUMULATO (finestra mobile di VINTAGE_WEEKS
        settimane) leggendo dalla Delta table (run precedente).
        La finestra arriva fino a max_date (ultimo giorno con Actual osservato).
        Fallback: seed Excel file per bootstrap iniziale.
        """
        # 1) Lettura Delta table (contiene output della run precedente)
        # v3.1 FIX: None SOLO se la tabella non esiste (primo run); ogni altro
        # errore viene rilanciato — un except generico azzerava silenziosamente
        # lo storico Forecast_Vintage.
        prev_df = kc.read_delta_or_none(spark, DELTA_TABLE)
        if prev_df is None:
            log.info("Lag-1 vintage: Delta table non trovata (primo run)")
            prev_df = pd.DataFrame()
        has_forecast = prev_df["Forecast"].notna().any() if "Forecast" in prev_df.columns else False

        # 2) Fallback a seed Excel se Delta non ha dati Forecast
        source = "delta"
        if prev_df.empty or not has_forecast:
            if SEED_FILE.exists():
                log.info(f"Lag-1 vintage: bootstrap da seed file {SEED_FILE.name}")
                prev_df = pd.read_excel(SEED_FILE, parse_dates=["ds"])
                source = "seed"
            else:
                log.info("Lag-1 vintage: nessun dato precedente — prima run")
                return None
        else:
            log.info("Lag-1 vintage: lettura da Delta table")

        prev_df["ds"] = pd.to_datetime(prev_df["ds"])

        # 3) Costruisci vintage accumulato: Forecast_Vintage (storico) + Forecast (lag-1)
        parts = []
        if "Forecast_Vintage" in prev_df.columns:
            hist = prev_df[["ds", "ID", "Forecast_Vintage"]].rename(columns={"Forecast_Vintage": "v"})
            parts.append(hist)
        if "Forecast" in prev_df.columns:
            new = prev_df[["ds", "ID", "Forecast"]].rename(columns={"Forecast": "v"})
            parts.append(new)

        if not parts:
            return None

        combined = pd.concat(parts, ignore_index=True)
        combined["v"] = pd.to_numeric(combined["v"], errors="coerce")
        combined = (
            combined.dropna(subset=["v"])
            .drop_duplicates(subset=["ds", "ID"], keep="first")
        )

        # 4) Filtra finestra mobile (fino a max_date, non split_date)
        window_start = current_max_date - pd.Timedelta(weeks=VINTAGE_WEEKS)
        combined = combined[(combined["ds"] <= current_max_date) & (combined["ds"] > window_start)]

        # 5) Se dopo il filtro non ci sono righe e il seed esiste, usa il seed (bootstrap)
        if combined.empty and SEED_FILE.exists() and source != "seed":
            log.info("Lag-1 vintage: Delta non ha dati nella finestra — fallback a seed file")
            prev_df = pd.read_excel(SEED_FILE, parse_dates=["ds"])
            prev_df["ds"] = pd.to_datetime(prev_df["ds"])
            parts = []
            if "Forecast_Vintage" in prev_df.columns:
                parts.append(prev_df[["ds", "ID", "Forecast_Vintage"]].rename(columns={"Forecast_Vintage": "v"}))
            if "Forecast" in prev_df.columns:
                parts.append(prev_df[["ds", "ID", "Forecast"]].rename(columns={"Forecast": "v"}))
            if parts:
                combined = pd.concat(parts, ignore_index=True)
                combined["v"] = pd.to_numeric(combined["v"], errors="coerce")
                combined = combined.dropna(subset=["v"]).drop_duplicates(subset=["ds", "ID"], keep="first")
                combined = combined[(combined["ds"] <= current_max_date) & (combined["ds"] > window_start)]

        log.info(
            f"Lag-1 vintage accumulato: {len(combined)} righe "
            f"({window_start.date()} → {current_max_date.date()}, finestra {VINTAGE_WEEKS} settimane)"
        )
        return combined.set_index(["ds", "ID"])["v"]

    lag1_vintage = _load_previous_forecast_vintage(max_date)

    merged_df = (
        df
        .merge(df_forecast[["ds", "ID", "Forecast", "Forecast_Lower", "Forecast_Upper"]], on=["ds", "ID"], how="outer")
        .rename(columns={"y": "Actual"})
    )
    merged_df["Forecast_Vintage"] = np.nan

    if lag1_vintage is not None:
        idx = pd.MultiIndex.from_arrays([merged_df["ds"], merged_df["ID"]])
        merged_df["Forecast_Vintage"] = lag1_vintage.reindex(idx).values

    merged_df = merged_df[kc.STANDARD_COLS]

log.info(f"merged_df: {merged_df.shape}")

# COMMAND ----------

# DBTITLE 1,10. Plot Time Series
# # =============================================================================
# # 10. PLOT — SERIE TEMPORALI
# # =============================================================================
# with timed("Generazione plot serie temporali"):
#     START_DATE = pd.Timestamp("2025-10-01")
#     STOP_DATE  = pd.Timestamp("2026-06-01")
#     N_COLS     = 4
#     BG_COLOR   = "#0b1c44"
#     SERIES_STYLE = {
#         "Actual":           dict(color="#76b3fa", marker="o", linestyle="-",  label="Actuals"),
#         "Forecast_Vintage": dict(color="#a07dfa", marker="x", linestyle="--", label="Forecast Vintage"),
#         "Forecast":         dict(color="#f7b267", marker="s", linestyle=":",  label="Forecast"),
#     }

#     plot_df = merged_df[merged_df["ds"].between(START_DATE, STOP_DATE)]
#     uids    = plot_df["ID"].unique()
#     n_rows  = int(np.ceil(len(uids) / N_COLS))

#     plt.style.use("dark_background")
#     fig, axes = plt.subplots(n_rows, N_COLS, figsize=(8 * N_COLS, 5 * n_rows))
#     axes = axes.flatten()

#     for i, uid in enumerate(uids):
#         ax     = axes[i]
#         subset = plot_df[plot_df["ID"] == uid]
#         ax.set_facecolor(BG_COLOR)
#         for col, style in SERIES_STYLE.items():
#             ax.plot(subset["ds"], subset[col], **style, alpha=0.8, linewidth=1.8)
#         ax.text(0.01, 0.95, uid, transform=ax.transAxes, color="white", fontsize=10,
#                 va="top", ha="left",
#                 bbox=dict(facecolor="#1a2a5a", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3"))
#         ax.grid(color="white", alpha=0.05)
#         ax.tick_params(axis="both", colors="white", labelsize=8)
#         ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
#         ax.legend().remove()

#     for ax in axes[len(uids):]:
#         fig.delaxes(ax)

#     fig.set_facecolor(BG_COLOR)
#     plt.tight_layout()
#     plot_file = PLOT_PATH / f"kelly_forecast_{RUN_ID}.png"
#     plt.savefig(plot_file, dpi=120, bbox_inches="tight")
#     plt.close()

# log.info(f"Plot salvato: {plot_file}")

# COMMAND ----------

# DBTITLE 1,11. Heatmap
# # =============================================================================
# # 11. PLOT — HEATMAP SETTIMANALE
# # =============================================================================
# with timed("Generazione heatmap"):
#     N_WEEKS = 6
#     df_plot = merged_df[~merged_df["ds"].dt.weekday.isin([5, 6])].copy()

#     start_date_plot = max_date + pd.offsets.Week(weekday=0)
#     end_date_plot   = start_date_plot + pd.Timedelta(weeks=N_WEEKS)

#     pivot_df = (
#         df_plot[df_plot["ds"].between(start_date_plot, end_date_plot, inclusive="left")]
#         .assign(week=lambda d: d["ds"].dt.isocalendar().week)
#         .groupby(["ID", "week"], as_index=False)["Forecast"]
#         .mean()
#         .pivot(index="ID", columns="week", values="Forecast")
#     )

#     total_row   = pd.DataFrame([[np.nan] * len(pivot_df.columns)], columns=pivot_df.columns, index=["Total"])
#     pivot_final = pd.concat([total_row, pivot_df]).astype(float)

#     fig, ax = plt.subplots(figsize=(8, 9))
#     heatmap = ax.imshow(pivot_final.values, aspect="auto", cmap="viridis")
#     plt.colorbar(heatmap, ax=ax, orientation="horizontal", pad=0.1, label="Forecast Assenteismo")
#     ax.set_xticks(range(len(pivot_final.columns)))
#     ax.set_xticklabels(pivot_final.columns)
#     ax.set_yticks(range(len(pivot_final.index)))
#     ax.set_yticklabels([r"$\bf{Total}$"] + list(pivot_final.index[1:]))
#     ax.set_xlabel("Week")
#     ax.set_title(f"Forecast Heatmap — da settimana {start_date_plot.date()}")

#     for (i, j), val in np.ndenumerate(pivot_final.values):
#         if not np.isnan(val):
#             ax.text(j, i, f"{val:.2f}", ha="center", va="center",
#                     color="red" if i == 0 else "white", fontsize=8)

#     ax.set_ylim(len(pivot_final) - 0.5, -0.2)
#     plt.tight_layout()
#     heatmap_file = PLOT_PATH / f"kelly_heatmap_{RUN_ID}.png"
#     plt.savefig(heatmap_file, dpi=120, bbox_inches="tight")
#     plt.close()

# log.info(f"Heatmap salvata: {heatmap_file}")

# COMMAND ----------

# DBTITLE 1,12. Metrics & Verdicts
# # =============================================================================
# # 12. METRICHE + VERDETTI PER ID (solo informativi — non bloccanti)
# # Finestra: ultimi 30 giorni di dati osservati (dove Forecast_Vintage è disponibile)
# # =============================================================================
# with timed("Calcolo metriche"):
#     eval_from = max_date - pd.Timedelta(days=30)
#     eval_mask = (
#         (merged_df["ds"] > eval_from) &
#         (merged_df["ds"] <= split_date) &
#         merged_df["Forecast_Vintage"].notna() &
#         (merged_df["Actual"] < 1)
#     )
#     df_eval = merged_df[eval_mask].copy()

#     def compute_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
#         err  = forecast - actual
#         num  = 2 * np.abs(err)
#         den  = np.abs(actual) + np.abs(forecast)
#         return {
#             "MAE":   round(float(np.mean(np.abs(err))), 4),
#             "Bias":  round(float(np.mean(err)), 4),
#             "RMSE":  round(float(np.sqrt(np.mean(err**2))), 4),
#             "SMAPE": round(float(np.mean(num[den != 0] / den[den != 0]) * 100), 2),
#             "N":     int(len(actual)),
#         }

#     if len(df_eval) >= 5:
#         global_metrics = compute_metrics(df_eval["Actual"], df_eval["Forecast_Vintage"])
#         log.info(
#             f"Metriche vintage (ultimi 30gg, N={global_metrics['N']}): "
#             f"MAE={global_metrics['MAE']}  Bias={global_metrics['Bias']}  "
#             f"RMSE={global_metrics['RMSE']}  SMAPE={global_metrics['SMAPE']}%"
#         )
#     else:
#         global_metrics = {"MAE": None, "Bias": None, "RMSE": None, "SMAPE": None, "N": len(df_eval)}
#         log.info("Metriche vintage: dati insufficienti (< 5 punti) — seconda run necessaria")

# # Verdetti per ID (informativi)
# GATE_THR = {
#     "wmae_pp_warn": 2.0, "wmae_pp_red": 3.0,
#     "bias_pp_min": -1.0, "bias_pp_max": 2.0,
#     "max_err_warn": 5.0, "max_err_red": 7.0,
#     "drift_red": 1.5,
#     "under_consec_red": 2,
# }

# per_id_verdicts: dict = {}

# if len(df_eval) >= 5:
#     eval_verd = df_eval[~df_eval["ds"].dt.weekday.isin([5, 6])].copy()
#     _iso = eval_verd["ds"].dt.isocalendar()
#     eval_verd["year_week"] = _iso["year"].astype(str) + "-W" + _iso["week"].astype(str).str.zfill(2)

#     for _uid in sorted(eval_verd["ID"].unique()):
#         _sub = eval_verd[eval_verd["ID"] == _uid].copy()
#         _sub["error_pp"]     = (_sub["Forecast_Vintage"] - _sub["Actual"]) * 100
#         _sub["abs_error_pp"] = _sub["error_pp"].abs()
#         _sub["under"]        = _sub["Forecast_Vintage"] < _sub["Actual"]

#         _mae_pp  = float(_sub["abs_error_pp"].mean())
#         _bias_pp = float(_sub["error_pp"].mean())
#         _rmse_pp = float(np.sqrt((_sub["error_pp"] ** 2).mean()))
#         _w       = np.where(_sub["under"], 2, 1)
#         _wmae_pp = float((_w * _sub["abs_error_pp"]).sum() / _w.sum())

#         _weekly = (
#             _sub.groupby("year_week")
#             .agg(Actual=("Actual", "mean"), Forecast=("Forecast_Vintage", "mean"))
#             .sort_index().reset_index()
#         )
#         _weekly["error_pp"]   = (_weekly["Forecast"] - _weekly["Actual"]) * 100
#         _weekly["abs_err_pp"] = _weekly["error_pp"].abs()
#         _weekly["under"]      = _weekly["Forecast"] < _weekly["Actual"]

#         _n_under = int(_weekly["under"].sum())
#         _n_weeks = len(_weekly)
#         _max_consec = _cur = 0
#         for _u in _weekly["under"]:
#             _cur = _cur + 1 if _u else 0
#             _max_consec = max(_max_consec, _cur)

#         _max_err_pp = float(_weekly["abs_err_pp"].max()) if _n_weeks > 0 else float("nan")
#         _drift_pp   = (float(_weekly.iloc[3]["abs_err_pp"] - _weekly.iloc[0]["abs_err_pp"])
#                        if _n_weeks >= 4 else float("nan"))

#         def _s3(val, warn, red):
#             return "RED" if val >= red else ("WARN" if val >= warn else "OK")

#         _wmae_s  = _s3(_wmae_pp, GATE_THR["wmae_pp_warn"], GATE_THR["wmae_pp_red"])
#         _bias_s  = ("OK"   if GATE_THR["bias_pp_min"] <= _bias_pp <= GATE_THR["bias_pp_max"]
#                     else "WARN" if -2 <= _bias_pp < GATE_THR["bias_pp_min"]
#                     else "RED")
#         _under_s = ("OK"  if _n_under <= 1
#                     else "RED"  if _max_consec >= GATE_THR["under_consec_red"]
#                     else "WARN")
#         _err_s   = (_s3(_max_err_pp, GATE_THR["max_err_warn"], GATE_THR["max_err_red"])
#                     if not np.isnan(_max_err_pp) else "OK")
#         _drift_s = ("RED" if not np.isnan(_drift_pp) and _drift_pp > GATE_THR["drift_red"]
#                     else "OK")

#         if _bias_s == "RED":
#             _verdict = "REVISE_MODEL_Bias"
#         elif any(_s == "RED" for _s in [_wmae_s, _under_s, _err_s, _drift_s]):
#             _failed  = [_n for _n, _s in [("WMAE", _wmae_s), ("Under", _under_s),
#                                            ("MaxErr", _err_s), ("Drift", _drift_s)] if _s == "RED"]
#             _verdict = f"REVISE_MODEL_{'_'.join(_failed)}"
#         elif any(_s == "WARN" for _s in [_wmae_s, _bias_s, _under_s, _err_s]):
#             _verdict = "MONITOR"
#         else:
#             _verdict = "PRODUCTION_READY"

#         per_id_verdicts[_uid] = {
#             "verdict": _verdict, "wmae_pp": round(_wmae_pp, 4), "bias_pp": round(_bias_pp, 4),
#             "mae_pp": round(_mae_pp, 4), "rmse_pp": round(_rmse_pp, 4),
#         }

#     n_revise  = sum(1 for v in per_id_verdicts.values() if v["verdict"].startswith("REVISE"))
#     n_monitor = sum(1 for v in per_id_verdicts.values() if v["verdict"] == "MONITOR")
#     n_ready   = sum(1 for v in per_id_verdicts.values() if v["verdict"] == "PRODUCTION_READY")
#     log.info(f"Verdetti per ID — REVISE: {n_revise}  MONITOR: {n_monitor}  READY: {n_ready}")
# else:
#     n_revise = n_monitor = n_ready = 0

# COMMAND ----------

# DBTITLE 1,13. Save Excel Output
# =============================================================================
# 13. SALVATAGGIO SU DELTA TABLE
# =============================================================================
OUTPUT_TABLE = "`sbx-logistics`.kelly.kelly_it_forecast"

with timed("Salvataggio su Delta table"):
    # v3.1: schema standard 7 colonne, round(4), overwrite + overwriteSchema
    _n_rows = kc.write_forecast_table(spark, merged_df, OUTPUT_TABLE)

log.info(f"✅ Tabella {OUTPUT_TABLE} salvata con {_n_rows} righe")

# COMMAND ----------

# DBTITLE 1,14. Run History CSV
# # =============================================================================
# # 14. HISTORY CSV (una riga per run — cresce settimana dopo settimana)
# # =============================================================================
# total_elapsed = round(time.perf_counter() - script_start, 1)
# _timings["TOTALE"] = total_elapsed

# history_row = pd.DataFrame([{
#     "run_id":               RUN_ID,
#     "run_datetime":         RUN_TS.strftime("%Y-%m-%d %H:%M:%S"),
#     "model_version":        MODEL_VERSION,
#     "last_obs_date":        str(max_date.date()),
#     "n_ids":                len(all_IDs),
#     "ids_excluded":         ", ".join(ids_to_exclude),
#     "eval_from":            str(eval_from.date()),
#     "eval_to":              str(split_date.date()),
#     "mae":                  global_metrics["MAE"],
#     "bias":                 global_metrics["Bias"],
#     "rmse":                 global_metrics["RMSE"],
#     "smape_pct":            global_metrics["SMAPE"],
#     "n_eval_obs":           global_metrics["N"],
#     "n_ids_revise":         n_revise,
#     "n_ids_monitor":        n_monitor,
#     "n_ids_ready":          n_ready,
#     "loss_current":         final_loss,
#     "n_epochs_current":     n_epochs,
#     "time_training_sec":    _timings.get("Training modello"),
#     "time_total_sec":       total_elapsed,
#     "output_file":          str(output_file),
#     "plot_file":            str(plot_file),
#     "heatmap_file":         str(heatmap_file),
#     "checkpoint_file":      str(LATEST_MODEL_PKL),
# }])

# if HISTORY_CSV.exists():
#     history_row.to_csv(HISTORY_CSV, mode="a", header=False, index=False)
# else:
#     history_row.to_csv(HISTORY_CSV, mode="w", header=True, index=False)

# log.info(f"History aggiornata: {HISTORY_CSV}")
# log.info(f"✅ Script completato in {total_elapsed}s.")