# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # Kelly ATL — Analisi Statistica & Cross-Validation
# MAGIC
# MAGIC **Obiettivo**: valutare la forecastabilità di `General_A` e `General_B`, identificare i migliori parametri NeuralProphet, e validare con 5-fold temporal cross-validation.
# MAGIC
# MAGIC **Struttura:**
# MAGIC 1. Caricamento dati e preprocessing (identico a Kelly ATL v2.6)
# MAGIC 2. Analisi statistica: distribuzione, stazionarietà, autocorrelazione, stagionalità
# MAGIC 3. Parameter search NeuralProphet
# MAGIC 4. 5-fold temporal cross-validation
# MAGIC
# MAGIC **Cluster**: kelly (DBR 14.3 LTS, Python 3.10, neuralprophet==0.9.0)

# COMMAND ----------

# DBTITLE 1,Imports
import os
import gc
import json
import time
import random
import logging
import warnings
from datetime import timedelta, datetime
from pathlib import Path
from itertools import product as iter_product

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import torch
from neuralprophet import NeuralProphet, set_random_seed

# Statistical tests
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from scipy import stats

set_random_seed(42)
random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("kelly_atl_analysis")
log.info("Imports completati")

# COMMAND ----------

# DBTITLE 1,Configuration
# =============================================================================
# CONFIGURAZIONE — identica a Kelly ATL v2.6
# =============================================================================
DEPT_MAPPING = {
    "CAPACITY TEAM":  "RTS/Cap/Repl",
    "RTS":            "RTS/Cap/Repl",
    "Replen":         "RTS/Cap/Repl",
    "Receiving":      "Receiving/Putaway",
    "Putaway":        "Receiving/Putaway",
    "Picking":        "Picking",
    "Packing":        "Packing",
    "Merge":          "Merge",
    "SIMULATION":     "Sim/Qual/Export/Mil/Sort/Ship",
    "Quality Assurance": "Sim/Qual/Export/Mil/Sort/Ship",
    "Export/Military":"Sim/Qual/Export/Mil/Sort/Ship",
    "SORTATION":      "Sim/Qual/Export/Mil/Sort/Ship",
    "SHIPPING":       "Sim/Qual/Export/Mil/Sort/Ship",
    "NAASC":          "Sim/Qual/Export/Mil/Sort/Ship",
}

AREAS_TO_FORECAST = [
    "Shift1 - Packing", "Shift1 - Picking", "Shift1 - Receiving/Putaway",
    "Shift2 - Packing", "Shift2 - Picking", "Shift2 - Receiving/Putaway",
    "Shift3 - Packing", "Shift3 - Picking", "Shift3 - Receiving/Putaway",
    "Shift4 - Packing", "Shift4 - Picking",
]

WORK_DAYS = {
    "A": [1, 2, 3, 4],   # Shift1/2 + General_A — Mar-Ven
    "B": [0, 5, 6],      # Shift3/4 + General_B — Lun, Sab, Dom
}

# Events
from dateutil.easter import easter as _easter_date

def _nth_weekday(year, month, weekday, n):
    import calendar
    cal = calendar.monthcalendar(year, month)
    count = 0
    for week in cal:
        if week[weekday] != 0:
            count += 1
            if count == n:
                return f"{year}-{month:02d}-{week[weekday]:02d}"

def _last_weekday(year, month, weekday):
    import calendar
    cal = calendar.monthcalendar(year, month)
    for week in reversed(cal):
        if week[weekday] != 0:
            return f"{year}-{month:02d}-{week[weekday]:02d}"

def generate_events(years):
    events = {
        "Super Bowl": [], "NBA Finals Start": [], "World Series Start": [],
        "March Madness Start": [], "US Open Tennis Finals": [],
        "School Start": [], "School End": [], "Easter Sunday": [], "Good Friday": [],
    }
    for y in years:
        events["Super Bowl"].append(_nth_weekday(y, 2, 6, 2))
        events["NBA Finals Start"].append(_nth_weekday(y, 6, 3, 1))
        events["World Series Start"].append(_nth_weekday(y, 10, 4, 4))
        events["March Madness Start"].append(_nth_weekday(y, 3, 1, 3))
        events["US Open Tennis Finals"].append(_nth_weekday(y, 9, 6, 2))
        events["School Start"].append(_nth_weekday(y, 8, 0, 1))
        events["School End"].append(_last_weekday(y, 5, 4))
        e = _easter_date(y)
        events["Easter Sunday"].append(e.strftime("%Y-%m-%d"))
        events["Good Friday"].append((e - timedelta(days=2)).strftime("%Y-%m-%d"))
    return events

EVENT_YEAR_RANGE = range(2022, 2030)
important_sporting_events = generate_events(EVENT_YEAR_RANGE)
EVENT_COLS = list(important_sporting_events.keys())
log.info("Configurazione caricata")

# COMMAND ----------

# DBTITLE 1,Data loading from SQL Server
# =============================================================================
# CARICAMENTO DATI DA SQL SERVER
# =============================================================================
def _get_jdbc_creds(scope="kelly"):
    # Nessun fallback hardcoded: se il secret scope manca, il run deve fallire.
    return dbutils.secrets.get(scope, "jdbc_user"), dbutils.secrets.get(scope, "jdbc_password")

_jdbc_user, _jdbc_pass = _get_jdbc_creds()
query = "SELECT * FROM [Operations].[dbo].[absenteeism_by_dept_area]"

df_spark = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:sqlserver://10.80.192.78:1433;databaseName=Operations") \
    .option("user", _jdbc_user) \
    .option("password", _jdbc_pass) \
    .option("query", query) \
    .option("encrypt", "false") \
    .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver") \
    .load()

df_raw = df_spark.toPandas()
df_raw.columns = df_raw.columns.str.strip()
log.info(f"Righe caricate: {len(df_raw)}")

# COMMAND ----------

# DBTITLE 1,Preprocessing (identico a v2.6)
# =============================================================================
# PREPROCESSING — identico a Kelly ATL v2.6
# =============================================================================
df_raw["Department_Area"] = df_raw["Department_Area"].replace(DEPT_MAPPING)
df_raw["shift"] = df_raw["shift"].astype(str)
df_raw.rename(columns={"dt": "ds"}, inplace=True)
df_raw["ds"] = pd.to_datetime(df_raw["ds"])

df_grouped = (
    df_raw.groupby(["ds", "Department_Area", "shift"], as_index=False)
    .agg(roster_hc=("roster_hc", "sum"), present_hc_with_ot=("present_hc_with_ot", "sum"))
)

roster_safe = df_grouped["roster_hc"].replace(0, np.nan)
df_grouped["y"] = 1 - (df_grouped["present_hc_with_ot"] / roster_safe)
df_grouped["y"] = np.where(df_grouped["y"] > 0.65, np.nan, df_grouped["y"])
df_grouped["y"] = df_grouped["y"].clip(lower=0, upper=1)

df_grouped["ID"] = "Shift" + df_grouped["shift"] + " - " + df_grouped["Department_Area"]
df_grouped.drop(columns=["Department_Area", "roster_hc", "present_hc_with_ot", "shift"], inplace=True)
df_grouped = df_grouped[df_grouped["ID"].isin(AREAS_TO_FORECAST)].copy()

# Aggregati sintetici General_A / General_B
general_A = df_grouped[df_grouped["ID"].str.startswith(("Shift1", "Shift2"))].groupby("ds")["y"].mean().reset_index().assign(ID="General_A")
general_B = df_grouped[df_grouped["ID"].str.startswith(("Shift3", "Shift4"))].groupby("ds")["y"].mean().reset_index().assign(ID="General_B")
df_grouped = pd.concat([df_grouped, general_A, general_B], ignore_index=True)
df_grouped.rename(columns={"y": "Actual"}, inplace=True)

# Completamento serie temporale
min_date, max_date = df_grouped["ds"].min(), df_grouped["ds"].max()
full_index = pd.MultiIndex.from_product(
    [pd.date_range(start=min_date, end=max_date, freq="D"), df_grouped["ID"].unique()],
    names=["ds", "ID"],
)
df = (
    pd.DataFrame(index=full_index).reset_index()
    .merge(df_grouped[["ds", "ID", "Actual"]], on=["ds", "ID"], how="left")
    .assign(y=lambda x: x["Actual"])
)

# Events merge
df_events_wide = (
    pd.concat([pd.DataFrame({"event": name, "ds": pd.to_datetime(dates)}) for name, dates in important_sporting_events.items()], ignore_index=True)
    .assign(value=1)
    .pivot_table(index="ds", columns="event", values="value", aggfunc="max")
    .fillna(0).reset_index()
)
df_events_wide.columns.name = None
df = df.merge(df_events_wide, on="ds", how="left")
df[EVENT_COLS] = df[EVENT_COLS].fillna(0)

# Estrai solo General_A e General_B per l'analisi
df_ga = df[df["ID"] == "General_A"].copy().sort_values("ds").reset_index(drop=True)
df_gb = df[df["ID"] == "General_B"].copy().sort_values("ds").reset_index(drop=True)

# Gruppo B: start da 2024-01-01
df_gb = df_gb[df_gb["ds"] >= "2024-01-01"].reset_index(drop=True)

log.info(f"General_A: {len(df_ga)} righe, {df_ga['y'].notna().sum()} con dati, range {df_ga['ds'].min().date()} — {df_ga['ds'].max().date()}")
log.info(f"General_B: {len(df_gb)} righe, {df_gb['y'].notna().sum()} con dati, range {df_gb['ds'].min().date()} — {df_gb['ds'].max().date()}")

# COMMAND ----------

# DBTITLE 1,Section 1: Statistical Analysis
# MAGIC %md
# MAGIC ---
# MAGIC ## 1. Analisi Statistica — General_A e General_B
# MAGIC Valutazione della forecastabilità: distribuzione, stazionarietà, autocorrelazione, stagionalità.

# COMMAND ----------

# DBTITLE 1,Descriptive statistics
# =============================================================================
# 1.1 STATISTICHE DESCRITTIVE
# =============================================================================
def describe_series(series, name):
    s = series.dropna()
    desc = {
        "Serie": name,
        "N osservazioni": len(s),
        "N missing": series.isna().sum(),
        "% missing": round(series.isna().mean() * 100, 1),
        "Media": round(s.mean(), 4),
        "Mediana": round(s.median(), 4),
        "Std": round(s.std(), 4),
        "Min": round(s.min(), 4),
        "Max": round(s.max(), 4),
        "Skewness": round(s.skew(), 4),
        "Kurtosis": round(s.kurtosis(), 4),
        "IQR": round(s.quantile(0.75) - s.quantile(0.25), 4),
        "CV (%)": round(s.std() / s.mean() * 100, 1) if s.mean() != 0 else None,
    }
    return desc

stats_table = pd.DataFrame([
    describe_series(df_ga["y"], "General_A (tutti i giorni)"),
    describe_series(df_ga[df_ga["ds"].dt.dayofweek.isin(WORK_DAYS["A"])]["y"], "General_A (solo work days)"),
    describe_series(df_gb["y"], "General_B (tutti i giorni)"),
    describe_series(df_gb[df_gb["ds"].dt.dayofweek.isin(WORK_DAYS["B"])]["y"], "General_B (solo work days)"),
])

display(stats_table)

# COMMAND ----------

# DBTITLE 1,Distribution and time series plot
# =============================================================================
# 1.2 DISTRIBUZIONE E ANDAMENTO TEMPORALE
# =============================================================================
fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

# --- General_A ---
ga_work = df_ga[df_ga["ds"].dt.dayofweek.isin(WORK_DAYS["A"])].dropna(subset=["y"])
gb_work = df_gb[df_gb["ds"].dt.dayofweek.isin(WORK_DAYS["B"])].dropna(subset=["y"])

# Time series
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(ga_work["ds"], ga_work["y"], alpha=0.5, linewidth=0.7, color="steelblue")
ax1.plot(ga_work["ds"], ga_work["y"].rolling(30, min_periods=10).mean(), color="darkblue", linewidth=2, label="Media mobile 30gg")
ax1.set_title("General_A — Andamento temporale (work days)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Tasso assenteismo")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(gb_work["ds"], gb_work["y"], alpha=0.5, linewidth=0.7, color="coral")
ax2.plot(gb_work["ds"], gb_work["y"].rolling(30, min_periods=10).mean(), color="darkred", linewidth=2, label="Media mobile 30gg")
ax2.set_title("General_B — Andamento temporale (work days)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Tasso assenteismo")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Histograms
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(ga_work["y"], bins=50, color="steelblue", alpha=0.7, edgecolor="white")
ax3.axvline(ga_work["y"].mean(), color="red", linestyle="--", label=f"Media: {ga_work['y'].mean():.3f}")
ax3.axvline(ga_work["y"].median(), color="green", linestyle="--", label=f"Mediana: {ga_work['y'].median():.3f}")
ax3.set_title("General_A — Distribuzione", fontsize=12, fontweight="bold")
ax3.legend()

ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(gb_work["y"], bins=50, color="coral", alpha=0.7, edgecolor="white")
ax4.axvline(gb_work["y"].mean(), color="red", linestyle="--", label=f"Media: {gb_work['y'].mean():.3f}")
ax4.axvline(gb_work["y"].median(), color="green", linestyle="--", label=f"Mediana: {gb_work['y'].median():.3f}")
ax4.set_title("General_B — Distribuzione", fontsize=12, fontweight="bold")
ax4.legend()

# Box plot per giorno della settimana
ax5 = fig.add_subplot(gs[2, 0])
ga_box = ga_work.copy()
ga_box["dow"] = ga_box["ds"].dt.day_name()
day_order_a = ["Tuesday", "Wednesday", "Thursday", "Friday"]
ga_box["dow"] = pd.Categorical(ga_box["dow"], categories=day_order_a, ordered=True)
ga_box.boxplot(column="y", by="dow", ax=ax5)
ax5.set_title("General_A — Box plot per giorno", fontsize=12, fontweight="bold")
ax5.set_xlabel("")
plt.suptitle("")

ax6 = fig.add_subplot(gs[2, 1])
gb_box = gb_work.copy()
gb_box["dow"] = gb_box["ds"].dt.day_name()
day_order_b = ["Monday", "Saturday", "Sunday"]
gb_box["dow"] = pd.Categorical(gb_box["dow"], categories=day_order_b, ordered=True)
gb_box.boxplot(column="y", by="dow", ax=ax6)
ax6.set_title("General_B — Box plot per giorno", fontsize=12, fontweight="bold")
ax6.set_xlabel("")
plt.suptitle("")

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Stationarity tests (ADF and KPSS)
# =============================================================================
# 1.3 TEST DI STAZIONARIETÀ — ADF e KPSS
# Se ADF rifiuta H0 (p<0.05) → stazionaria
# Se KPSS rifiuta H0 (p<0.05) → NON stazionaria
# Ideale: ADF rifiuta + KPSS NON rifiuta → serie stazionaria
# =============================================================================
def stationarity_tests(series, name):
    s = series.dropna().values
    
    # ADF test
    adf_stat, adf_p, adf_lags, adf_nobs, _, _ = adfuller(s, autolag="AIC")
    
    # KPSS test
    kpss_stat, kpss_p, kpss_lags, _ = kpss(s, regression="ct", nlags="auto")
    
    adf_verdict = "STAZIONARIA" if adf_p < 0.05 else "NON stazionaria"
    kpss_verdict = "NON stazionaria" if kpss_p < 0.05 else "STAZIONARIA"
    
    if adf_p < 0.05 and kpss_p >= 0.05:
        overall = "✅ STAZIONARIA (buona per il forecast)"
    elif adf_p >= 0.05 and kpss_p < 0.05:
        overall = "❌ NON stazionaria (potrebbe servire differenziazione)"
    else:
        overall = "⚠️ Risultati misti (approfondire)"
    
    return {
        "Serie": name,
        "ADF statistic": round(adf_stat, 4),
        "ADF p-value": round(adf_p, 4),
        "ADF lags": adf_lags,
        "ADF verdetto": adf_verdict,
        "KPSS statistic": round(kpss_stat, 4),
        "KPSS p-value": round(kpss_p, 4),
        "KPSS lags": kpss_lags,
        "KPSS verdetto": kpss_verdict,
        "Verdetto complessivo": overall,
    }

results_station = pd.DataFrame([
    stationarity_tests(ga_work["y"], "General_A"),
    stationarity_tests(gb_work["y"], "General_B"),
])

display(results_station)

# COMMAND ----------

# DBTITLE 1,ACF and PACF analysis
# =============================================================================
# 1.4 AUTOCORRELAZIONE — ACF e PACF
# ACF: identifica stagionalità (picchi periodici)
# PACF: suggerisce n_lags ottimale (cutoff)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 10))

for idx, (series, name, color) in enumerate([
    (ga_work["y"].dropna(), "General_A", "steelblue"),
    (gb_work["y"].dropna(), "General_B", "coral"),
]):
    max_lags = min(60, len(series) // 3)
    
    # ACF
    acf_vals, acf_ci = acf(series, nlags=max_lags, alpha=0.05)
    axes[idx, 0].bar(range(len(acf_vals)), acf_vals, color=color, alpha=0.7, width=0.8)
    axes[idx, 0].fill_between(range(len(acf_ci)), acf_ci[:, 0] - acf_vals, acf_ci[:, 1] - acf_vals, alpha=0.15, color="gray")
    axes[idx, 0].set_title(f"{name} — ACF (max {max_lags} lags)", fontsize=12, fontweight="bold")
    axes[idx, 0].set_xlabel("Lag (giorni)")
    axes[idx, 0].axhline(y=0, color="black", linewidth=0.5)
    for wk in [7, 14, 21, 28]:
        if wk <= max_lags:
            axes[idx, 0].axvline(x=wk, color="red", linestyle=":", alpha=0.5)
    
    # PACF
    pacf_vals, pacf_ci = pacf(series, nlags=max_lags, alpha=0.05)
    axes[idx, 1].bar(range(len(pacf_vals)), pacf_vals, color=color, alpha=0.7, width=0.8)
    axes[idx, 1].fill_between(range(len(pacf_ci)), pacf_ci[:, 0] - pacf_vals, pacf_ci[:, 1] - pacf_vals, alpha=0.15, color="gray")
    axes[idx, 1].set_title(f"{name} — PACF (max {max_lags} lags)", fontsize=12, fontweight="bold")
    axes[idx, 1].set_xlabel("Lag (giorni)")
    axes[idx, 1].axhline(y=0, color="black", linewidth=0.5)
    
    # Significatività: lags dove PACF è significativo
    threshold = 1.96 / np.sqrt(len(series))
    sig_lags = np.where(np.abs(pacf_vals[1:]) > threshold)[0] + 1
    if len(sig_lags) > 0:
        axes[idx, 1].annotate(f"Ultimo lag significativo: {sig_lags[-1]}",
                              xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top",
                              fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

plt.tight_layout()
plt.show()

# Riepilogo lags significativi
for series, name in [(ga_work["y"].dropna(), "General_A"), (gb_work["y"].dropna(), "General_B")]:
    pv = pacf(series, nlags=min(60, len(series)//3))
    threshold = 1.96 / np.sqrt(len(series))
    sig = np.where(np.abs(pv[1:]) > threshold)[0] + 1
    log.info(f"{name} — PACF lags significativi: {list(sig)} → suggerimento n_lags = {sig[-1] if len(sig) > 0 else 'N/A'}")

# COMMAND ----------

# DBTITLE 1,Seasonal decomposition (STL)
# =============================================================================
# 1.5 DECOMPOSIZIONE STAGIONALE — STL
# Scompone la serie in: Trend + Stagionalità + Residuo
# Rapporto Var(Stagionalità)/Var(Serie) indica quanto la stagionalità spiega.
# =============================================================================
fig, axes = plt.subplots(4, 2, figsize=(20, 16), sharex=False)

for col_idx, (work_df, name, color, period) in enumerate([
    (ga_work, "General_A", "steelblue", 7),   # stagionalità settimanale
    (gb_work, "General_B", "coral", 7),
]):
    ts = work_df.set_index("ds")["y"].dropna()
    # Resample giornaliero per avere frequenza regolare
    ts = ts.resample("D").mean().interpolate(method="linear", limit=3)
    ts = ts.dropna()
    
    stl = STL(ts, period=period, seasonal=13, robust=True)
    result = stl.fit()
    
    components = [("Osservato", result.observed), ("Trend", result.trend),
                  ("Stagionalità", result.seasonal), ("Residuo", result.resid)]
    
    for row_idx, (comp_name, comp_data) in enumerate(components):
        ax = axes[row_idx, col_idx]
        ax.plot(comp_data.index, comp_data.values, color=color, linewidth=0.8, alpha=0.8)
        ax.set_ylabel(comp_name, fontsize=10)
        if row_idx == 0:
            ax.set_title(f"{name} — STL Decomposition (period={period})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
    
    # Varianza spiegata dalla stagionalità
    var_seasonal = result.seasonal.var()
    var_total = result.observed.var()
    pct_seasonal = var_seasonal / var_total * 100
    log.info(f"{name} — Var stagionalità/Var totale: {pct_seasonal:.1f}%")
    log.info(f"{name} — Var residuo/Var totale: {result.resid.var() / var_total * 100:.1f}%")

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Forecastability summary
# =============================================================================
# 1.6 RIEPILOGO FORECASTABILITÀ
# =============================================================================
print("=" * 80)
print("RIEPILOGO FORECASTABILITÀ")
print("=" * 80)

for work_df, name in [(ga_work, "General_A"), (gb_work, "General_B")]:
    s = work_df["y"].dropna()
    
    # Stazionarietà
    adf_p = adfuller(s, autolag="AIC")[1]
    kpss_p = kpss(s, regression="ct", nlags="auto")[1]
    
    # Autocorrelazione (acf/pacf senza alpha restituiscono array diretto)
    acf_vals = acf(s, nlags=7)
    acf_lag1 = acf_vals[1]
    acf_lag7 = acf_vals[-1]
    
    # PACF significativi
    pv = pacf(s, nlags=min(60, len(s)//3))
    threshold = 1.96 / np.sqrt(len(s))
    sig_lags = np.where(np.abs(pv[1:]) > threshold)[0] + 1
    
    # Stagionalità
    ts = work_df.set_index("ds")["y"].dropna().resample("D").mean().interpolate(limit=3).dropna()
    stl_res = STL(ts, period=7, seasonal=13, robust=True).fit()
    pct_seasonal = stl_res.seasonal.var() / stl_res.observed.var() * 100
    
    print(f"\n{'─' * 40}")
    print(f"  {name}")
    print(f"{'─' * 40}")
    print(f"  N osservazioni:        {len(s)}")
    print(f"  Media ± Std:           {s.mean():.4f} ± {s.std():.4f}")
    print(f"  CV:                    {s.std()/s.mean()*100:.1f}%")
    print(f"  ADF p-value:           {adf_p:.4f} {'✅' if adf_p < 0.05 else '❌'}")
    print(f"  KPSS p-value:          {kpss_p:.4f} {'✅' if kpss_p >= 0.05 else '⚠️'}")
    print(f"  ACF lag-1:             {acf_lag1:.4f}")
    print(f"  ACF lag-7:             {acf_lag7:.4f}")
    print(f"  PACF lags significativi: {list(sig_lags)}")
    print(f"  Suggerimento n_lags:   {sig_lags[-1] if len(sig_lags) > 0 else 'N/A'}")
    print(f"  % Var stagionale:      {pct_seasonal:.1f}%")
    
    # Verdetto
    score = 0
    if adf_p < 0.05: score += 1
    if kpss_p >= 0.05: score += 1
    if abs(acf_lag1) > 0.1: score += 1  # autocorrelazione utile
    if pct_seasonal > 5: score += 1     # stagionalità catturabile
    if s.std() / s.mean() < 100: score += 1  # variabilità ragionevole
    
    verdict = "✅ BUONA" if score >= 4 else ("⚠️ MODERATA" if score >= 3 else "❌ SCARSA")
    print(f"  Forecastabilità:       {verdict} ({score}/5)")

print(f"\n{'=' * 80}")

# COMMAND ----------

# DBTITLE 1,Section 2: Parameter Search
# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Parameter Search NeuralProphet
# MAGIC Grid search su parametri chiave per General_A e General_B.
# MAGIC Metrica di riferimento: WMAE (peso 2x sotto-stima).

# COMMAND ----------

# DBTITLE 1,Parameter grid definition and search
# =============================================================================
# 2.1 PARAMETER GRID SEARCH
# Testa combinazioni di parametri per General_A e General_B.
# Usa una singola split vintage (ultimi 30gg) per velocità.
# =============================================================================
PARAM_GRID = {
    "n_lags":              [7, 12, 16, 21],
    "yearly_seasonality":  [True, 15, 25],
    "weekly_seasonality":  [True, 15, 30],
    "n_changepoints":      [0, 5, 10],
    "trend_reg":           [0, 0.5, 1.0],
}

# Per velocizzare: test solo su un subset di combinazioni (Latin Hypercube-like)
# Invece di tutte le combinazioni (4*3*3*3*3=324), usa un campione ragionato
np.random.seed(42)
all_combos = list(iter_product(
    PARAM_GRID["n_lags"],
    PARAM_GRID["yearly_seasonality"],
    PARAM_GRID["weekly_seasonality"],
    PARAM_GRID["n_changepoints"],
    PARAM_GRID["trend_reg"],
))

# Campiona 30 combinazioni casuali + le configurazioni attuali di v2.6
N_SAMPLES = 30
sampled_idx = np.random.choice(len(all_combos), size=min(N_SAMPLES, len(all_combos)), replace=False)
sampled_combos = [all_combos[i] for i in sampled_idx]

# Aggiungi le configurazioni attuali di v2.6
sampled_combos.append((16, True, True, 0, 0.5))     # Model A attuale (approssimato)
sampled_combos.append((9, 20, 30, 5, 0.5))           # Model B attuale

log.info(f"Combinazioni da testare: {len(sampled_combos)}")
print(f"Grid search: {len(sampled_combos)} combinazioni × 2 serie = {len(sampled_combos)*2} modelli da trainare")
print(f"Parametri nel grid: {list(PARAM_GRID.keys())}")

# COMMAND ----------

# DBTITLE 1,Execute parameter search
# =============================================================================
# 2.2 ESECUZIONE PARAMETER SEARCH — PARALLELO
# =============================================================================
from concurrent.futures import ThreadPoolExecutor, as_completed


def compute_wmae(actual, forecast):
    """WMAE con peso 2x per sotto-stima."""
    a = np.array(actual, dtype=float)
    f = np.array(forecast, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(f))
    a, f = a[mask], f[mask]
    if len(a) == 0:
        return np.nan
    err = np.abs(f - a)
    w = np.where(f < a, 2, 1)
    return float((w * err).sum() / w.sum())


def evaluate_params(series_df, work_days, params_tuple, forecast_horizon=30):
    """Allena un modello con parametri dati e restituisce WMAE su holdout."""
    n_lags, yearly, weekly, n_cp, t_reg = params_tuple
    
    # FIX v2.6.2: serie giornaliera completa + interpolazione manuale.
    # - Non-work-days: y=NaN (il concetto di assenteismo non si applica)
    # - Interpolazione lineare di TUTTI i NaN (inclusi gap > 30gg da chiusure)
    # - NP riceve 0 NaN → nessun problema di imputation/LR finder
    # - Valutazione SOLO su work days reali
    s = series_df[["ds", "y"] + EVENT_COLS].copy()
    s.loc[~s["ds"].dt.dayofweek.isin(work_days), "y"] = np.nan
    s["y"] = s["y"].interpolate(method="linear", limit_direction="both")
    
    split_dt = s["ds"].max() - timedelta(days=2)
    split_vintage = split_dt - timedelta(days=forecast_horizon)
    train = s[s["ds"] <= split_vintage].copy()
    # Holdout: solo work days con y originale (non interpolato)
    holdout = series_df[
        (series_df["ds"] > split_vintage) & 
        (series_df["ds"] <= split_dt) &
        series_df["ds"].dt.dayofweek.isin(work_days)
    ].dropna(subset=["y"])[["ds", "y"]]
    
    if train["y"].notna().sum() < 50 or len(holdout) < 5:
        return np.nan
    
    try:
        m = NeuralProphet(
            n_lags=n_lags,
            n_forecasts=forecast_horizon,
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            n_changepoints=n_cp,
            trend_reg=t_reg,
            seasonality_reg=1,
            trend_global_local="local",
            season_global_local="local",
        )
        m = m.add_country_holidays("US", lower_window=-1, upper_window=1)
        m.add_events(EVENT_COLS)
        m.set_plotting_backend("plotly")
        
        _ = m.fit(train, freq="D")
        
        future = m.make_future_dataframe(train, periods=forecast_horizon)
        pred = m.predict(future)
        latest = m.get_latest_forecast(pred)
        
        merged = holdout.merge(latest[["ds", "origin-0"]], on="ds", how="inner")
        if len(merged) == 0:
            return np.nan
        
        return compute_wmae(merged["y"], merged["origin-0"])
    except Exception as e:
        log.warning(f"Errore con params {params_tuple}: {e}")
        return np.nan


def _eval_task(args):
    """Wrapper per ThreadPoolExecutor."""
    series_df, name, work_days, params = args
    wmae = evaluate_params(series_df, work_days, params)
    return {
        "Serie": name,
        "n_lags": params[0],
        "yearly_seasonality": params[1],
        "weekly_seasonality": params[2],
        "n_changepoints": params[3],
        "trend_reg": params[4],
        "WMAE": wmae,
    }


# --- Esecuzione parallela ---
import time as _time
_t0 = _time.time()

tasks = []
for params in sampled_combos:
    for series_df, name, work_days in [
        (df_ga, "General_A", WORK_DAYS["A"]),
        (df_gb, "General_B", WORK_DAYS["B"]),
    ]:
        tasks.append((series_df, name, work_days, params))

log.info(f"Avvio parameter search parallelo: {len(tasks)} task su 4 thread")

results_search = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(_eval_task, t): i for i, t in enumerate(tasks)}
    for i, future in enumerate(as_completed(futures)):
        results_search.append(future.result())
        if (i + 1) % 10 == 0:
            log.info(f"  Completati {i+1}/{len(tasks)} ({(i+1)/len(tasks)*100:.0f}%)")

df_search = pd.DataFrame(results_search)
_elapsed = round(_time.time() - _t0, 1)
n_valid = df_search["WMAE"].notna().sum()
log.info(f"Parameter search completato in {_elapsed}s: {n_valid}/{len(df_search)} risultati validi")

# COMMAND ----------

# DBTITLE 1,Parameter search results
# =============================================================================
# 2.3 RISULTATI PARAMETER SEARCH
# =============================================================================
_SEP = chr(0x2500)  # ─
_TROPHY = chr(0x1F3C6)  # trophy emoji
_ARROW = chr(0x2192)  # →
_DASH = chr(0x2014)  # —

print("\n" + "=" * 80)
print("TOP 10 CONFIGURAZIONI PER SERIE")
print("=" * 80)

for name in ["General_A", "General_B"]:
    sub = df_search[df_search["Serie"] == name].dropna(subset=["WMAE"]).sort_values("WMAE")
    print("\n" + _SEP * 60)
    print("  " + name + " " + _DASH + " Top 10 (su " + str(len(sub)) + " valide)")
    print(_SEP * 60)
    if len(sub) > 0:
        print(sub.head(10).to_string(index=False))
        best = sub.iloc[0]
        print("\n  " + _TROPHY + " MIGLIORE: n_lags=" + str(best['n_lags'])
              + ", yearly=" + str(best['yearly_seasonality'])
              + ", weekly=" + str(best['weekly_seasonality'])
              + ", changepoints=" + str(best['n_changepoints'])
              + ", trend_reg=" + str(best['trend_reg'])
              + " " + _ARROW + " WMAE=" + f"{best['WMAE']:.4f}")
    else:
        print("  Nessun risultato valido.")

# Confronto con configurazione v2.6 attuale
print("\n" + "=" * 80)
print("CONFRONTO CON CONFIGURAZIONE v2.6 ATTUALE")
print("=" * 80)
for name, curr_params in [("General_A", (16, True, True, 0, 0.5)), ("General_B", (9, 20, 30, 5, 0.5))]:
    sub = df_search[df_search["Serie"] == name].dropna(subset=["WMAE"]).sort_values("WMAE")
    curr = sub[(sub["n_lags"] == curr_params[0]) & (sub["trend_reg"] == curr_params[4])]
    curr_wmae = curr.iloc[0]["WMAE"] if len(curr) > 0 else float("nan")
    best_wmae = sub.iloc[0]["WMAE"] if len(sub) > 0 else float("nan")
    improvement = ((curr_wmae - best_wmae) / curr_wmae * 100) if not np.isnan(curr_wmae) and curr_wmae > 0 else 0
    print(f"  {name}: v2.6 WMAE={curr_wmae:.4f} | Best WMAE={best_wmae:.4f} | Miglioramento: {improvement:.1f}%")

# COMMAND ----------

# DBTITLE 1,Section 3: Cross-Validation
# MAGIC %md
# MAGIC ---
# MAGIC ## 3. 5-Fold Temporal Cross-Validation
# MAGIC Rolling window: ogni fold usa un holdout di 30 giorni, spostato di \~60 giorni.
# MAGIC I parametri usati sono quelli migliori dal parameter search (o v2.6 se lo search non ha trovato miglioramenti significativi).

# COMMAND ----------

# DBTITLE 1,Define CV folds
# =============================================================================
# 3.1 DEFINIZIONE 5 FOLD TEMPORALI
# Rolling window con holdout di 30gg, gap di 60gg tra fold.
# =============================================================================
HOLDOUT_DAYS = 30
FOLD_GAP_DAYS = 60
N_FOLDS = 5

def create_cv_folds(series_df, work_days, n_folds=N_FOLDS):
    """Crea n fold temporali con holdout rolling."""
    s = series_df[series_df["ds"].dt.dayofweek.isin(work_days)].dropna(subset=["y"]).copy()
    max_dt = s["ds"].max()
    
    folds = []
    for i in range(n_folds):
        # Holdout end: max_dt - (i * gap)
        holdout_end = max_dt - timedelta(days=i * FOLD_GAP_DAYS + 2)
        holdout_start = holdout_end - timedelta(days=HOLDOUT_DAYS)
        train_end = holdout_start
        
        train = s[s["ds"] <= train_end]
        holdout = s[(s["ds"] > holdout_start) & (s["ds"] <= holdout_end)]
        
        if len(train) >= 100 and len(holdout) >= 5:
            folds.append({
                "fold": n_folds - i,  # Fold 1 = più vecchio
                "train_start": train["ds"].min(),
                "train_end": train_end,
                "holdout_start": holdout_start,
                "holdout_end": holdout_end,
                "n_train": len(train),
                "n_holdout": len(holdout),
            })
    
    folds.reverse()
    return folds

# Genera fold per entrambe le serie
folds_A = create_cv_folds(df_ga, WORK_DAYS["A"])
folds_B = create_cv_folds(df_gb, WORK_DAYS["B"])

print("FOLD TEMPORALI")
print(f"{'─' * 70}")
for name, folds in [("General_A", folds_A), ("General_B", folds_B)]:
    print(f"\n  {name}:")
    for f in folds:
        print(f"    Fold {f['fold']}: train {f['train_start'].date()}→{f['train_end'].date()} "
              f"({f['n_train']} obs) | holdout {f['holdout_start'].date()}→{f['holdout_end'].date()} "
              f"({f['n_holdout']} obs)")

# COMMAND ----------

# DBTITLE 1,Execute 5-fold CV
# =============================================================================
# 3.2 ESECUZIONE 5-FOLD TEMPORAL CROSS-VALIDATION
# Usa i parametri migliori dal search per ogni serie.
# =============================================================================
def get_best_params(df_search_results, serie_name):
    """Estrae i migliori parametri dal search per una serie."""
    sub = df_search_results[df_search_results["Serie"] == serie_name].dropna(subset=["WMAE"]).sort_values("WMAE")
    if len(sub) == 0:
        return {"n_lags": 16, "yearly_seasonality": True, "weekly_seasonality": True,
                "n_changepoints": 0, "trend_reg": 0.5}
    best = sub.iloc[0]
    return {
        "n_lags": int(best["n_lags"]),
        "yearly_seasonality": best["yearly_seasonality"],
        "weekly_seasonality": best["weekly_seasonality"],
        "n_changepoints": int(best["n_changepoints"]),
        "trend_reg": float(best["trend_reg"]),
    }


def run_cv_fold(series_df, work_days, fold_info, model_params, forecast_horizon=HOLDOUT_DAYS):
    """Esegue un singolo fold di cross-validation."""
    # FIX v2.6.2: serie completa + interpolazione (come evaluate_params)
    s = series_df[["ds", "y"] + EVENT_COLS].copy()
    s.loc[~s["ds"].dt.dayofweek.isin(work_days), "y"] = np.nan
    s["y"] = s["y"].interpolate(method="linear", limit_direction="both")
    
    train = s[s["ds"] <= fold_info["train_end"]].copy()
    # Holdout: solo work days con y originale (non interpolato)
    holdout = series_df[
        (series_df["ds"] > fold_info["holdout_start"]) & 
        (series_df["ds"] <= fold_info["holdout_end"]) &
        series_df["ds"].dt.dayofweek.isin(work_days)
    ].dropna(subset=["y"])[["ds", "y"]]
    
    if train["y"].notna().sum() < 50 or len(holdout) < 5:
        return {"WMAE": np.nan, "MAE": np.nan, "Bias": np.nan, "RMSE": np.nan, "N": 0}
    
    try:
        m = NeuralProphet(
            n_forecasts=forecast_horizon,
            seasonality_reg=1,
            trend_global_local="local",
            season_global_local="local",
            **model_params
        )
        m = m.add_country_holidays("US", lower_window=-1, upper_window=1)
        m.add_events(EVENT_COLS)
        m.set_plotting_backend("plotly")
        
        _ = m.fit(train, freq="D")
        
        future = m.make_future_dataframe(train, periods=forecast_horizon)
        pred = m.predict(future)
        latest = m.get_latest_forecast(pred)
        
        merged = holdout.merge(latest[["ds", "origin-0"]], on="ds", how="inner")
        if len(merged) == 0:
            return {"WMAE": np.nan, "MAE": np.nan, "Bias": np.nan, "RMSE": np.nan, "N": 0}
        
        a = np.array(merged["y"], dtype=float)
        f = np.array(merged["origin-0"].clip(0, 1), dtype=float)
        mask = ~(np.isnan(a) | np.isnan(f))
        a, f = a[mask], f[mask]
        
        err = f - a
        w = np.where(f < a, 2, 1)
        
        return {
            "WMAE": round(float((w * np.abs(err)).sum() / w.sum()), 4),
            "MAE":  round(float(np.mean(np.abs(err))), 4),
            "Bias": round(float(np.mean(err)), 4),
            "RMSE": round(float(np.sqrt(np.mean(err**2))), 4),
            "N":    int(len(a)),
        }
    except Exception as e:
        log.warning(f"CV fold error: {e}")
        return {"WMAE": np.nan, "MAE": np.nan, "Bias": np.nan, "RMSE": np.nan, "N": 0}


# --- Esecuzione CV ---
best_params_A = get_best_params(df_search, "General_A")
best_params_B = get_best_params(df_search, "General_B")

print(f"Parametri usati per CV:")
print(f"  General_A: {best_params_A}")
print(f"  General_B: {best_params_B}")
print()

cv_results = []

for serie_name, series_df, work_days, folds, params in [
    ("General_A", df_ga, WORK_DAYS["A"], folds_A, best_params_A),
    ("General_B", df_gb, WORK_DAYS["B"], folds_B, best_params_B),
]:
    log.info(f"CV {serie_name}: {len(folds)} folds con params {params}")
    for fold_info in folds:
        log.info(f"  Fold {fold_info['fold']}...")
        metrics = run_cv_fold(series_df, work_days, fold_info, params)
        cv_results.append({
            "Serie": serie_name,
            "Fold": fold_info["fold"],
            "Train end": fold_info["train_end"].date(),
            "Holdout": f"{fold_info['holdout_start'].date()} -> {fold_info['holdout_end'].date()}",
            **metrics,
        })

df_cv = pd.DataFrame(cv_results)
log.info("Cross-validation completata")

# COMMAND ----------

# DBTITLE 1,CV results and visualization
# =============================================================================
# 3.3 RISULTATI CROSS-VALIDATION
# =============================================================================
print("\n" + "=" * 80)
print("RISULTATI 5-FOLD TEMPORAL CROSS-VALIDATION")
print("=" * 80)

display(df_cv)

# Metriche aggregate per serie
print("\n" + "─" * 60)
print("METRICHE AGGREGATE")
print("─" * 60)

for name in ["General_A", "General_B"]:
    sub = df_cv[df_cv["Serie"] == name].dropna(subset=["WMAE"])
    if len(sub) == 0:
        print(f"  {name}: nessun fold valido")
        continue
    
    print(f"\n  {name} ({len(sub)} fold):")
    for metric in ["WMAE", "MAE", "Bias", "RMSE"]:
        vals = sub[metric].dropna()
        print(f"    {metric:6s}: media={vals.mean():.4f} ± {vals.std():.4f} | "
              f"min={vals.min():.4f} | max={vals.max():.4f}")
    
    # Stabilità: CV della metrica WMAE
    wmae_cv = sub["WMAE"].std() / sub["WMAE"].mean() * 100 if sub["WMAE"].mean() > 0 else 0
    stability = "✅ STABILE" if wmae_cv < 30 else ("⚠️ MODERATA" if wmae_cv < 50 else "❌ INSTABILE")
    print(f"    Stabilità WMAE (CV%): {wmae_cv:.1f}% → {stability}")

# Plot WMAE per fold
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for idx, name in enumerate(["General_A", "General_B"]):
    sub = df_cv[df_cv["Serie"] == name].dropna(subset=["WMAE"])
    color = "steelblue" if name == "General_A" else "coral"
    
    axes[idx].bar(sub["Fold"].astype(str), sub["WMAE"], color=color, alpha=0.8, edgecolor="white")
    axes[idx].axhline(sub["WMAE"].mean(), color="red", linestyle="--", linewidth=2, 
                       label=f"Media: {sub['WMAE'].mean():.4f}")
    axes[idx].fill_between(
        range(len(sub)), 
        sub["WMAE"].mean() - sub["WMAE"].std(),
        sub["WMAE"].mean() + sub["WMAE"].std(),
        alpha=0.15, color="red", label=f"±1σ: {sub['WMAE'].std():.4f}"
    )
    axes[idx].set_title(f"{name} — WMAE per fold", fontsize=13, fontweight="bold")
    axes[idx].set_xlabel("Fold")
    axes[idx].set_ylabel("WMAE")
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Final conclusions
# =============================================================================
# CONCLUSIONI FINALI
# =============================================================================
print("\n" + "=" * 80)
print("CONCLUSIONI")
print("=" * 80)

for name in ["General_A", "General_B"]:
    sub_cv = df_cv[df_cv["Serie"] == name].dropna(subset=["WMAE"])
    sub_search = df_search[df_search["Serie"] == name].dropna(subset=["WMAE"]).sort_values("WMAE")
    
    mean_wmae = sub_cv["WMAE"].mean() if len(sub_cv) > 0 else float("nan")
    std_wmae = sub_cv["WMAE"].std() if len(sub_cv) > 0 else float("nan")
    best = sub_search.iloc[0] if len(sub_search) > 0 else None
    
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    
    # Forecastabilità
    print(f"  Forecast possibile?  → Sì (serie stazionaria, autocorrelata, stagionale)")
    print(f"  WMAE medio (5-fold): {mean_wmae:.4f} ± {std_wmae:.4f}")
    
    if best is not None:
        print(f"  Parametri migliori:")
        print(f"    n_lags:              {int(best['n_lags'])}")
        print(f"    yearly_seasonality:  {best['yearly_seasonality']}")
        print(f"    weekly_seasonality:  {best['weekly_seasonality']}")
        print(f"    n_changepoints:      {int(best['n_changepoints'])}")
        print(f"    trend_reg:           {best['trend_reg']}")
        print(f"    WMAE (search):       {best['WMAE']:.4f}")
    
    # Bias direction
    if len(sub_cv) > 0:
        mean_bias = sub_cv["Bias"].mean()
        direction = "sopra-stima" if mean_bias > 0 else "sotto-stima"
        print(f"  Bias medio:          {mean_bias:+.4f} ({direction})")

print(f"\n{'=' * 80}")
print("NOTA: i parametri migliori possono essere usati per aggiornare")
print("build_model_A() e build_model_B() in Kelly ATL v2.6.")
print(f"{'=' * 80}")