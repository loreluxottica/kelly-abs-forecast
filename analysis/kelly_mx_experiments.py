# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # Tijuana — Feature Experiment Notebook (DAILY)
# MAGIC
# MAGIC **Versione:** 1.1 — Frequenza Giornaliera  
# MAGIC **Basato su:** Columbus_Experiments_v1.1  
# MAGIC **Data:** 2026-04-29
# MAGIC
# MAGIC ## Obiettivo
# MAGIC Valutare l'impatto di 4 famiglie di feature sul MAE del validation window (**3 finestre stagionali di ~30 giorni**).  
# MAGIC Ogni esperimento cambia **una sola variabile** rispetto al baseline, seed fisso = 42.  
# MAGIC **Frequenza giornaliera** — NeuralProphet gestisce la zero-inflation nativamente (`drop_missing=True`).
# MAGIC
# MAGIC | ID | Variabile testata |
# MAGIC |----|-------------------|
# MAGIC | E0 | **Baseline** (MX federal holidays ±1, custom events: Carnaval, Semana Santa, School BC, Fiestas Patrias) |
# MAGIC | E1 | Holiday window **±2** invece di ±1 |
# MAGIC | E2 | **Senza School events** (rimuove School_Start_BC e School_End_BC) |
# MAGIC | E3 | **Senza custom events** (solo MX federal holidays) |
# MAGIC | E4 | +**Buen Fin** (Black Friday messicano, ultima settimana di novembre) |
# MAGIC
# MAGIC ## Metrica principale: MAE sulle 3 finestre stagionali
# MAGIC - **3 validation windows** (~30 giorni): Estate 2025, Inverno 2025, Primavera 2026
# MAGIC - Modello: NeuralProphet daily, `n_lags=14`, `weekly_seasonality=5`, `n_forecasts=30`
# MAGIC - Esecuzione parallelizzata con ThreadPoolExecutor
# MAGIC - Metriche: MAE (principale), Bias, RMSE

# COMMAND ----------

# DBTITLE 1,Section 0
# MAGIC %md
# MAGIC ## 0. Imports & Config

# COMMAND ----------

# DBTITLE 1,Install neuralprophet
# MAGIC %pip install neuralprophet

# COMMAND ----------

# DBTITLE 1,Imports
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch, functools

# PyTorch 2.6+: override torch.load per weights_only=False
_orig = torch.serialization.load

@functools.wraps(_orig)
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig(*args, **kwargs)

torch.load = _patched_load

from neuralprophet import NeuralProphet, set_random_seed

print('Libraries loaded (torch.load patched).')

# COMMAND ----------

# DBTITLE 1,Configuration
# ── JDBC ──
JDBC_URL  = "jdbc:sqlserver://10.80.192.78:1433;databaseName=Business_Intelligence"
JDBC_USER = dbutils.secrets.get(scope="kelly", key="jdbc_user")
JDBC_PWD  = dbutils.secrets.get(scope="kelly", key="jdbc_password")

JDBC_QUERY = """
    SELECT
        [Numero]  AS Clerk,
        [Turno]   AS [Shift],
        [Fecha]   AS [Date],
        12        AS TotalHours,
        CASE
            WHEN [absenteeism] > 12 THEN 12
            ELSE [absenteeism]
        END       AS AbsHours
    FROM [dbo].[MX03_HeadcountData_Timestamps]
    WHERE CAST([Tipo_de_Dia] AS VARCHAR(max)) = 'Hábil'
      AND [year] IN ('2024', '2025', '2026')
"""

# ── Forecast (DAILY / Business Days) ──
TARGET_SHIFTS    = ['A', 'B', 'C', 'D']
START_DATE       = pd.Timestamp('2024-01-01')
FORECAST_DAYS    = 22     # ~1 mese di business days
FREQ             = 'B'    # Business day frequency

# ── Output ──
VOLUME_BASE = "/Volumes/jdawave/kelly_mx/kelly_mx_volume"
OUTPUT_PATH = Path(f"{VOLUME_BASE}/output")
try:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
except OSError:
    print(f"⚠ Volume non montato: {OUTPUT_PATH}")

SEED = 42
set_random_seed(SEED)
print('Configuration OK (DAILY / Business Days).')
print(f'  Frequency: {FREQ}')
print(f'  Forecast horizon: {FORECAST_DAYS} business days (~1 mese)')

# COMMAND ----------

# DBTITLE 1,Section 1
# MAGIC %md
# MAGIC ## 1. Data Pipeline (run once)
# MAGIC
# MAGIC JDBC → Spark → pandas → aggregazione settimanale per turno.

# COMMAND ----------

# DBTITLE 1,JDBC Load
spark_df = (
    spark.read.format('jdbc')
    .option('url',      JDBC_URL + ';encrypt=true;trustServerCertificate=true')
    .option('dbtable',  f'({JDBC_QUERY}) AS subq')
    .option('user',     JDBC_USER)
    .option('password', JDBC_PWD)
    .option('driver',   'com.microsoft.sqlserver.jdbc.SQLServerDriver')
    .load()
)
df_raw = spark_df.toPandas()
df_raw['Date']     = pd.to_datetime(df_raw['Date'], errors='coerce')
df_raw['AbsHours'] = pd.to_numeric(df_raw['AbsHours'], errors='coerce')
df_raw['TotalHours'] = pd.to_numeric(df_raw['TotalHours'], errors='coerce')

print(f'Righe caricate: {len(df_raw):,}')
print(f'Periodo: {df_raw["Date"].min().date()} -> {df_raw["Date"].max().date()}')
print(f'Turni:   {sorted(df_raw["Shift"].unique())}')

# COMMAND ----------

# DBTITLE 1,Clean and weekly aggregation
df_raw['Absenteeism'] = df_raw['AbsHours'] / df_raw['TotalHours'].replace(0, np.nan)

# Filtro turni validi
df = df_raw[
    (df_raw['Shift'].isin(TARGET_SHIFTS)) &
    (df_raw['AbsHours'] >= 0) &
    (df_raw['Absenteeism'] <= 1)
].copy()
print(f'Righe dopo filtro turni: {len(df):,}')

# Aggregazione GIORNALIERA per turno (no weekly)
daily = (
    df.groupby(['Date', 'Shift'], as_index=False)
    .agg(
        AbsHours   = ('AbsHours',   'sum'),
        TotalHours = ('TotalHours', 'sum'),
        Headcount  = ('Clerk',      'nunique')
    )
)
daily['y'] = daily['AbsHours'] / daily['TotalHours']
daily = daily.rename(columns={'Date': 'ds', 'Shift': 'ID'})
daily = daily[daily['ds'] >= START_DATE].sort_values(['ds', 'ID']).reset_index(drop=True)

max_date = daily['ds'].max()
min_date = daily['ds'].min()

print(f'\nDataset giornaliero: {min_date.date()} -> {max_date.date()}')
print(f'Turni: {sorted(daily["ID"].unique())}')
print(f'Giorni per turno:')
print(daily.groupby('ID')['ds'].count().to_string())
print(f'\nStatistiche y:')
print(f'  Zero: {(daily["y"] == 0).sum():,} ({(daily["y"] == 0).mean():.1%})')
print(f'  Mean: {daily["y"].mean():.4f}')

# COMMAND ----------

# DBTITLE 1,Complete time series
# Completamento giorni mancanti — solo BUSINESS DAYS (feriali)
# I dati fonte filtrano Tipo_de_Dia='Hábil' → no weekend nell'indice
all_bdays = pd.bdate_range(start=min_date, end=max_date, freq='B')
full_index = pd.MultiIndex.from_product(
    [all_bdays, TARGET_SHIFTS],
    names=['ds', 'ID']
)

df_base = (
    pd.DataFrame(index=full_index)
    .reset_index()
    .merge(daily[['ds', 'ID', 'y']], on=['ds', 'ID'], how='left')
)
df_base = df_base[['ds', 'ID', 'y']].sort_values(['ds', 'ID']).reset_index(drop=True)

# Anni per holidays
years = sorted(df_base['ds'].dt.year.unique().tolist())
years = years + [max(years) + 1]

print(f'Dataset completo (business days): {df_base.shape}')
print(f'Periodo: {df_base["ds"].min().date()} -> {df_base["ds"].max().date()}')
print(f'y non-NaN: {df_base["y"].notna().sum():,}')
print(f'y NaN:     {df_base["y"].isna().sum():,} ({df_base["y"].isna().mean():.1%})')
print(f'Business days totali: {len(all_bdays)}')

# COMMAND ----------

# DBTITLE 1,Section 2
# MAGIC %md
# MAGIC ## 2. Definizione Esperimenti
# MAGIC
# MAGIC Eventi custom specifici per Tijuana / Baja California.  
# MAGIC A frequenza daily le date vengono usate direttamente (NeuralProphet gestisce window internamente).

# COMMAND ----------

# DBTITLE 1,Experiment definitions
# --- Date eventi custom ---
CARNAVAL_TIJUANA = ['2024-02-10', '2025-03-01', '2026-02-14', '2027-02-06']
SEMANA_SANTA     = ['2024-03-28', '2025-04-17', '2026-04-02', '2027-03-25']
SCHOOL_START_BC  = ['2024-08-26', '2025-08-25', '2026-08-24', '2027-08-23']
SCHOOL_END_BC    = ['2024-07-12', '2025-07-11', '2026-07-10', '2027-07-09']
FIESTAS_PATRIAS  = ['2024-09-16', '2025-09-16', '2026-09-16', '2027-09-16']
# Buen Fin: Black Friday messicano, terzo venerdì di novembre
BUEN_FIN         = ['2024-11-15', '2025-11-21', '2026-11-20', '2027-11-19']

BASE_CUSTOM = {
    'Carnaval_Tijuana': CARNAVAL_TIJUANA,
    'Semana_Santa':     SEMANA_SANTA,
    'School_Start_BC':  SCHOOL_START_BC,
    'School_End_BC':    SCHOOL_END_BC,
    'Fiestas_Patrias':  FIESTAS_PATRIAS,
}

EXPERIMENTS = [
    {
        'id':    'E0_baseline',
        'label': 'E0 \u2014 Baseline',
        'holiday_window': (-1, 1),
        'custom_events':  dict(BASE_CUSTOM),
    },
    {
        'id':    'E1_window2',
        'label': 'E1 \u2014 Holiday window \u00b12',
        'holiday_window': (-2, 2),
        'custom_events':  dict(BASE_CUSTOM),
    },
    {
        'id':    'E2_no_school',
        'label': 'E2 \u2014 Senza School events',
        'holiday_window': (-1, 1),
        'custom_events':  {
            'Carnaval_Tijuana': CARNAVAL_TIJUANA,
            'Semana_Santa':     SEMANA_SANTA,
            'Fiestas_Patrias':  FIESTAS_PATRIAS,
        },
    },
    {
        'id':    'E3_no_custom',
        'label': 'E3 \u2014 Solo MX holidays',
        'holiday_window': (-1, 1),
        'custom_events':  {},
    },
    {
        'id':    'E4_buen_fin',
        'label': 'E4 \u2014 +Buen Fin',
        'holiday_window': (-1, 1),
        'custom_events':  {**BASE_CUSTOM, 'Buen_Fin': BUEN_FIN},
    },
]

print(f'{len(EXPERIMENTS)} esperimenti definiti.')
for e in EXPERIMENTS:
    n_ev = len(e['custom_events'])
    hw   = e['holiday_window']
    print(f'  {e["id"]}: window={hw}, custom_events={n_ev}')

# COMMAND ----------

# DBTITLE 1,Section 3
# MAGIC %md
# MAGIC ## 3. Experiment Runner (DAILY)
# MAGIC
# MAGIC Modello giornaliero: `freq='D'`, `n_lags=14`, `yearly_seasonality=8`, `weekly_seasonality=5`.  
# MAGIC Eventi custom alla data esatta — NeuralProphet applica `lower_window`/`upper_window` internamente.  
# MAGIC **3 validation windows** (~30gg): Estate, Inverno, Primavera — esecuzione parallela.

# COMMAND ----------

# DBTITLE 1,Metrics & Runner
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools, logging

# Suppress verbose NeuralProphet logging
logging.getLogger('NP').setLevel(logging.WARNING)
logging.getLogger('NP.config').setLevel(logging.WARNING)
logging.getLogger('NP.data').setLevel(logging.WARNING)

def calc_metrics(actual, forecast):
    mask = actual.notna() & forecast.notna()
    a = actual[mask].to_numpy(dtype=float)
    f = forecast[mask].to_numpy(dtype=float)
    if len(a) == 0:
        return dict(n=0, MAE=np.nan, Bias=np.nan, RMSE=np.nan)
    return dict(
        n    = len(a),
        MAE  = round(float(np.mean(np.abs(f - a))), 4),
        Bias = round(float(np.mean(f - a)), 4),
        RMSE = round(float(np.sqrt(np.mean((f - a)**2))), 4),
    )


# ── 3 Validation Windows (~22 business days = ~1 mese) ──
VAL_WINDOWS = {
    'Estate':    ('2025-06-02', '2025-06-30'),
    'Inverno':   ('2025-12-01', '2025-12-31'),
    'Primavera': ('2026-03-02', '2026-03-31'),
}


def run_experiment_daily(cfg, df_base, years, val_start_str, val_end_str):
    """Esegue un esperimento daily (business days, no AR) su una finestra di validazione."""
    hw_lo, hw_hi = cfg['holiday_window']
    custom_evts  = cfg['custom_events']
    val_start = pd.Timestamp(val_start_str)
    val_end   = pd.Timestamp(val_end_str)

    # Prepara eventi custom (date esatte)
    df_exp = df_base.copy()
    ec = []
    evts_wide = None

    if custom_evts:
        evts_long = pd.concat([
            pd.DataFrame({'event': name, 'ds': pd.to_datetime(dates)})
            for name, dates in custom_evts.items()
        ], ignore_index=True).drop_duplicates()
        evts_wide = (
            evts_long.assign(value=1)
            .pivot_table(index='ds', columns='event', values='value', aggfunc='max')
            .fillna(0).reset_index()
        )
        df_exp = df_exp.merge(evts_wide, on='ds', how='left')
        ec = list(custom_evts.keys())
        df_exp[ec] = df_exp[ec].fillna(0).astype(int)

    # Train: tutto prima della finestra di validazione
    train_v = df_exp[df_exp['ds'] < val_start].copy()

    # Build model (DAILY, decomposition — no AR)
    # n_lags=0: modello puro trend+seasonality+events, gestisce dati sparsi
    set_random_seed(SEED)
    m = NeuralProphet(
        n_changepoints      = 5,
        trend_reg           = 0.5,
        trend_global_local  = 'local',
        seasonality_mode    = 'additive',
        yearly_seasonality  = 8,
        weekly_seasonality  = 5,       # pattern DOW
        daily_seasonality   = False,
        seasonality_reg     = 1,
        season_global_local = 'local',
        drop_missing        = True,
    )
    m = m.add_country_holidays('MX', lower_window=hw_lo, upper_window=hw_hi)
    if ec:
        m.add_events(ec, lower_window=hw_lo, upper_window=hw_hi)

    # Fit
    try:
        metrics = m.fit(train_v, freq=FREQ)
        final_loss = round(float(metrics['Loss'].iloc[-1]), 6)
    except Exception as e:
        return {'id': cfg['id'], 'label': cfg['label'],
                'global': dict(n=0, MAE=np.nan, Bias=np.nan, RMSE=np.nan),
                'per_serie': {}, 'final_loss': np.nan, 'error': str(e)}

    # Predict
    try:
        future_v = m.make_future_dataframe(train_v, periods=FORECAST_DAYS)
        if ec and evts_wide is not None:
            cols_drop = [c for c in ec if c in future_v.columns]
            future_v = future_v.drop(columns=cols_drop, errors='ignore')
            future_v = future_v.merge(evts_wide, on='ds', how='left')
            future_v[ec] = future_v[ec].fillna(0).astype(int)

        forecast_v = m.predict(future_v)
    except Exception as e:
        return {'id': cfg['id'], 'label': cfg['label'],
                'global': dict(n=0, MAE=np.nan, Bias=np.nan, RMSE=np.nan),
                'per_serie': {}, 'final_loss': final_loss, 'error': str(e)}

    # Estrai previsioni (yhat1 con modello senza AR)
    preds = forecast_v[
        (forecast_v['ds'] >= val_start) & (forecast_v['ds'] <= val_end) &
        forecast_v['yhat1'].notna()
    ][['ds', 'ID', 'yhat1']].rename(columns={'yhat1': 'Forecast'})
    preds['Forecast'] = preds['Forecast'].clip(0, 1)

    # Eval: merge con actual nella finestra
    eval_df = df_exp[
        (df_exp['ds'] >= val_start) & (df_exp['ds'] <= val_end)
    ][['ds', 'ID', 'y']].rename(columns={'y': 'Actual'})
    eval_df = eval_df.merge(preds, on=['ds', 'ID'], how='inner')
    eval_df = eval_df.dropna(subset=['Actual', 'Forecast'])

    # Metriche
    global_m = calc_metrics(eval_df['Actual'], eval_df['Forecast'])
    per_serie = {}
    for uid in sorted(eval_df['ID'].unique()):
        sub = eval_df[eval_df['ID'] == uid]
        per_serie[uid] = calc_metrics(sub['Actual'], sub['Forecast'])

    return {
        'id':         cfg['id'],
        'label':      cfg['label'],
        'global':     global_m,
        'per_serie':  per_serie,
        'final_loss': final_loss,
    }


print('Runner definito (DAILY, decomposition, business days).')
print(f'  Model: freq={FREQ}, n_lags=0 (no AR), weekly_seasonality=5')
print(f'  Componenti: trend + yearly + weekly + MX holidays + custom events')
print(f'  Validation: {len(VAL_WINDOWS)} windows x ~22 bdays')
for k, (s, e) in VAL_WINDOWS.items():
    print(f'    {k}: {s} → {e}')

# COMMAND ----------

# DBTITLE 1,Section 4
# MAGIC %md
# MAGIC ## 4. Esecuzione Esperimenti (Daily, Parallelo)
# MAGIC
# MAGIC > 5 esperimenti × 3 finestre stagionali = 15 run, parallelizzati con ThreadPoolExecutor.  
# MAGIC > Ogni run: train su dati precedenti alla finestra, eval sulla finestra (~30 giorni).

# COMMAND ----------

# Quick check dataset daily
print(f'Shape: {df_base.shape}')
print(f'Colonne: {df_base.columns.tolist()}')
print(f'\nPrime 5 righe:')
display(df_base.head(10))

# COMMAND ----------

# DBTITLE 1,Run all experiments
# ═══ ESECUZIONE PARALLELA: 5 esperimenti × 3 finestre = 15 run ═══

tasks = list(itertools.product(EXPERIMENTS, VAL_WINDOWS.items()))
results_multi = {}  # {(exp_id, window_name): result}

print(f'{len(EXPERIMENTS)} esperimenti × {len(VAL_WINDOWS)} finestre = {len(tasks)} run')
print(f'Lancio in parallelo (max_workers=4)...\n')

with ThreadPoolExecutor(max_workers=4) as executor:
    future_map = {}
    for cfg, (wname, (ws, we)) in tasks:
        fut = executor.submit(run_experiment_daily, cfg, df_base, years, ws, we)
        future_map[fut] = (cfg['id'], wname)

    done_count = 0
    for fut in as_completed(future_map):
        exp_id, wname = future_map[fut]
        done_count += 1
        try:
            res = fut.result()
            results_multi[(exp_id, wname)] = res
            mae = res['global']['MAE']
            print(f'  [{done_count}/{len(tasks)}] {exp_id} | {wname}: MAE={mae}')
        except Exception as e:
            print(f'  [{done_count}/{len(tasks)}] {exp_id} | {wname}: ERRORE — {e}')
            results_multi[(exp_id, wname)] = None

print(f'\n✓ Completati {sum(1 for v in results_multi.values() if v is not None)}/{len(tasks)} run.')

# COMMAND ----------

# DBTITLE 1,Section 5
# MAGIC %md
# MAGIC ## 5. Risultati — Tabella Comparativa (Daily)

# COMMAND ----------

# DBTITLE 1,Global summary table
# ═══ RISULTATI AGGREGATI (media MAE su 3 stagioni) ═══

# Per finestra
print('=' * 70)
print('RISULTATI PER FINESTRA TEMPORALE')
print('=' * 70)
for wname in VAL_WINDOWS:
    print(f'\n--- {wname} ({VAL_WINDOWS[wname][0]} → {VAL_WINDOWS[wname][1]}) ---')
    print(f'{"Esperimento":<20} {"MAE":>8} {"Bias":>8} {"RMSE":>8} {"n":>6}')
    print('-' * 54)
    for cfg in EXPERIMENTS:
        r = results_multi.get((cfg['id'], wname))
        if r and r['global']['MAE'] is not None:
            g = r['global']
            print(f'{cfg["id"]:<20} {g["MAE"]:>8.4f} {g["Bias"]:>8.4f} {g["RMSE"]:>8.4f} {g["n"]:>6}')
        else:
            print(f'{cfg["id"]:<20} {"ERR":>8}')

# Aggregato
print(f'\n{"=" * 70}')
print('RANKING AGGREGATO (media MAE su 3 stagioni)')
print(f'{"=" * 70}')

agg_rows = []
for cfg in EXPERIMENTS:
    maes, biases, rmses = [], [], []
    for wname in VAL_WINDOWS:
        r = results_multi.get((cfg['id'], wname))
        if r and r['global']['MAE'] is not None:
            maes.append(r['global']['MAE'])
            biases.append(r['global']['Bias'])
            rmses.append(r['global']['RMSE'])
    agg_rows.append({
        'Exp': cfg['id'], 'Label': cfg['label'],
        'Avg_MAE': round(np.mean(maes), 4) if maes else np.nan,
        'Avg_Bias': round(np.mean(biases), 4) if biases else np.nan,
        'Avg_RMSE': round(np.mean(rmses), 4) if rmses else np.nan,
        'n_windows': len(maes),
    })

summary = pd.DataFrame(agg_rows).set_index('Exp')
baseline_avg = summary.loc['E0_baseline', 'Avg_MAE']
summary['Delta_MAE'] = (summary['Avg_MAE'] - baseline_avg).round(4)
summary['Delta%'] = ((summary['Avg_MAE'] - baseline_avg) / baseline_avg * 100).round(1)
summary = summary.sort_values('Avg_MAE')

print(summary.to_string())
print(f'\nBaseline Avg MAE: {baseline_avg}')
print(f'\n🏆 Vincitore: {summary.index[0]} — Avg MAE {summary["Avg_MAE"].iloc[0]:.4f} ({summary["Delta%"].iloc[0]:+.1f}% vs baseline)')

# COMMAND ----------

# DBTITLE 1,Per-serie tables
# Per-turno: media MAE su 3 finestre per ogni turno
series_ids = TARGET_SHIFTS

print('\n=== MAE medio per turno (3 finestre) ===')
shift_rows = []
for cfg in EXPERIMENTS:
    row = {'Exp': cfg['id']}
    for sid in series_ids:
        shift_maes = []
        for wname in VAL_WINDOWS:
            r = results_multi.get((cfg['id'], wname))
            if r and sid in r.get('per_serie', {}):
                m = r['per_serie'][sid].get('MAE')
                if m is not None:
                    shift_maes.append(m)
        row[sid] = round(np.mean(shift_maes), 4) if shift_maes else np.nan
    shift_rows.append(row)

df_shifts = pd.DataFrame(shift_rows).set_index('Exp')
print(df_shifts.to_string())

# Delta vs baseline per turno
print('\n=== Delta MAE% vs baseline per turno ===')
bl_row = df_shifts.loc['E0_baseline']
df_delta = ((df_shifts - bl_row) / bl_row * 100).round(1)
print(df_delta.to_string())

# COMMAND ----------

# DBTITLE 1,Section 6
# MAGIC %md
# MAGIC ## 6. Esperimenti Combinati + Ranking Finale

# COMMAND ----------

# DBTITLE 1,Section Combined
# MAGIC %md
# MAGIC ## 6b. Esperimento Combinato (Daily)
# MAGIC
# MAGIC Basandoci sui risultati individuali, testiamo le combinazioni vincenti:
# MAGIC - **E5** = E1 + E2 + E4: ±2, no School, +Buen Fin
# MAGIC - **E6** = E1 + E3: ±2, no custom events
# MAGIC - **E7** = E1 + E3 + E4: ±2, solo Buen Fin
# MAGIC - **E8** = E1 + E2: ±2, no School (Carnaval + Semana Santa + Fiestas Patrias)
# MAGIC
# MAGIC Esecuzione parallela sulle stesse 3 finestre stagionali.

# COMMAND ----------

# DBTITLE 1,Run combined experiment
# ═══ ESPERIMENTI COMBINATI (DAILY) ═══

COMBINED_EXPERIMENTS = [
    {
        'id':    'E5_combined',
        'label': 'E5 — ±2, no School, +Buen Fin',
        'holiday_window': (-2, 2),
        'custom_events': {
            'Carnaval_Tijuana': CARNAVAL_TIJUANA,
            'Semana_Santa':     SEMANA_SANTA,
            'Fiestas_Patrias':  FIESTAS_PATRIAS,
            'Buen_Fin':         BUEN_FIN,
        },
    },
    {
        'id':    'E6_no_custom_w2',
        'label': 'E6 — ±2, no custom (E1+E3)',
        'holiday_window': (-2, 2),
        'custom_events':  {},
    },
    {
        'id':    'E7_only_buenfin_w2',
        'label': 'E7 — ±2, solo Buen Fin (E1+E3+E4)',
        'holiday_window': (-2, 2),
        'custom_events':  {'Buen_Fin': BUEN_FIN},
    },
    {
        'id':    'E8_noSchool_w2',
        'label': 'E8 — ±2, no School (E1+E2)',
        'holiday_window': (-2, 2),
        'custom_events': {
            'Carnaval_Tijuana': CARNAVAL_TIJUANA,
            'Semana_Santa':     SEMANA_SANTA,
            'Fiestas_Patrias':  FIESTAS_PATRIAS,
        },
    },
]

tasks_comb = list(itertools.product(COMBINED_EXPERIMENTS, VAL_WINDOWS.items()))
print(f'{len(COMBINED_EXPERIMENTS)} combinazioni × {len(VAL_WINDOWS)} finestre = {len(tasks_comb)} run')
print('Lancio in parallelo...\n')

with ThreadPoolExecutor(max_workers=4) as executor:
    future_map = {}
    for cfg, (wname, (ws, we)) in tasks_comb:
        fut = executor.submit(run_experiment_daily, cfg, df_base, years, ws, we)
        future_map[fut] = (cfg['id'], wname)

    done_count = 0
    for fut in as_completed(future_map):
        exp_id, wname = future_map[fut]
        done_count += 1
        try:
            res = fut.result()
            results_multi[(exp_id, wname)] = res
            print(f'  [{done_count}/{len(tasks_comb)}] {exp_id} | {wname}: MAE={res["global"]["MAE"]}')
        except Exception as e:
            print(f'  [{done_count}/{len(tasks_comb)}] {exp_id} | {wname}: ERRORE — {e}')
            results_multi[(exp_id, wname)] = None

# ── RANKING FINALE COMPLETO (9 esperimenti) ──
ALL_EXPERIMENTS = EXPERIMENTS + COMBINED_EXPERIMENTS

print(f'\n{"=" * 70}')
print('RANKING FINALE COMPLETO (media MAE su 3 stagioni)')
print(f'{"=" * 70}')

agg_final = []
for cfg in ALL_EXPERIMENTS:
    maes = []
    for wname in VAL_WINDOWS:
        r = results_multi.get((cfg['id'], wname))
        if r and r['global']['MAE'] is not None:
            maes.append(r['global']['MAE'])
    agg_final.append({
        'Exp': cfg['id'], 'Label': cfg['label'],
        'Avg_MAE': round(np.mean(maes), 4) if maes else np.nan,
        'n_windows': len(maes),
    })

df_final = pd.DataFrame(agg_final).set_index('Exp')
bl_avg = df_final.loc['E0_baseline', 'Avg_MAE']
df_final['Delta_MAE'] = (df_final['Avg_MAE'] - bl_avg).round(4)
df_final['Delta%'] = ((df_final['Avg_MAE'] - bl_avg) / bl_avg * 100).round(1)
df_final = df_final.sort_values('Avg_MAE')
print(df_final.to_string())

winner = df_final.index[0]
print(f'\n{"=" * 70}')
print(f'🏆 VINCITORE ASSOLUTO: {winner}')
print(f'   {df_final.loc[winner, "Label"]}')
print(f'   Avg MAE: {df_final.loc[winner, "Avg_MAE"]:.4f} ({df_final.loc[winner, "Delta%"]:+.1f}% vs baseline)')
print(f'{"=" * 70}')