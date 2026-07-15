# Databricks notebook source
# MAGIC %md
# MAGIC # Columbus DC ? Absenteeism Forecast Pipeline
# MAGIC **Version:** 1.3  
# MAGIC **Freq:** Daily | **Target:** Abs_rate per Dept ? Shift  
# MAGIC **Model:** NeuralProphet (global model, local trend/seasonality)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Changelog
# MAGIC
# MAGIC ### v1.5 — 2026-07-09 (vintage bounds)
# MAGIC - **Congelati anche i bound**: `Forecast_Vintage_Lower` / `Forecast_Vintage_Upper` con la stessa
# MAGIC   logica lag-1 del point — permette di misurare la copertura empirica del PI 90% (schema a 9 colonne).
# MAGIC
# MAGIC ### v1.4 — 2026-07-08 (schema standard + intervalli di previsione)
# MAGIC - **Prediction interval 90%**: `quantiles=[0.05, 0.95]` nel modello; nuove colonne
# MAGIC   `Forecast_Lower` / `Forecast_Upper` nella Delta table (schema standard 7 colonne per tutte le geografie).
# MAGIC - **Modulo condiviso `common/kelly_common.py`**: metriche, estrazione forecast+quantili,
# MAGIC   carry-forward vintage, scrittura Delta standardizzata (round 4 decimali, prima assente).
# MAGIC - **Fix dedup overlap (bug)**: aggiunto `drop_duplicates` su (CostCenter, Shift, Date) dopo il concat
# MAGIC   historical+2026 — prima le righe sovrapposte venivano SOMMATE due volte dal groupby
# MAGIC   (il commento "df26 vince sull'overlap" era falso per un'aggregazione sum). Stessa logica di Dallas.
# MAGIC - **Fix vintage read**: `except Exception` sostituito da gestione esplicita table-not-found —
# MAGIC   un errore transiente non azzera piu silenziosamente lo storico Forecast_Vintage.
# MAGIC
# MAGIC ### v1.3 ? 2026-04-29
# MAGIC - **HPT-optimized** (Columbus_HPT_v1.0.ipynb ? 100 trial Bayesian search, 3-fold temporal CV):
# MAGIC   CV score baseline 0.0757 ? best 0.0667 (**?11.9%**); fold3_spring MAE 0.0380 ? 0.0348 (+8.4%).
# MAGIC - `loss_func`: Huber ? **MAE** (import. fANOVA 0.255 ? allineamento diretto con metrica di eval).
# MAGIC - `yearly_seasonality`: 10 ? **6** (import. 0.280 ? 3 anni di dati non giustificano Fourier alto; era overfitting).
# MAGIC - `learning_rate`: 0.003 ? **0.00190** (import. 0.293 ? optimizer pi? cauto e stabile).
# MAGIC - `n_changepoints`: 5 ? **12** (trend pi? flessibile; `trend_reg` abbassato da 0.3 a 0.071).
# MAGIC - `seasonality_reg`: 0.3 ? **0.159**; `epochs`: auto ? **99** (fisso per riproducibilit?).
# MAGIC - `weekly_seasonality`: invariato a **4** (import. 0.047 ? non significativo).
# MAGIC - Miglioramento concentrato su ECP Weekend Shift (MAE ?0.0324, ?34.6%);
# MAGIC   lieve regressione su Lens Weekend (+0.0099) e ECP 2nd (+0.0035): da monitorare.
# MAGIC - Bias summer (fold1_summer ?5.5%) strutturale, non risolvibile con HPT.
# MAGIC
# MAGIC ### v1.2 ? 2026-04-17
# MAGIC - **+Ramadan (inizio+fine)**: aggiunti Ramadan_Start e Ramadan_End come eventi custom (?1 giorno).
# MAGIC   Miglioramento MAE globale -1.8% (0.0387 ? 0.0380), consistente su 5/8 serie;
# MAGIC   picco sul Weekend Shift ECP -4.0%. Incluso per meccanismo causale plausibile
# MAGIC   (forza lavoro con presenza di lavoratori musulmani).
# MAGIC - **OH State Holidays**: testato, zero effetto (Ohio = US federal). Non incluso.
# MAGIC - **Holiday window ?2**: testato, nessun miglioramento rispetto a ?1. Non incluso.
# MAGIC - **School events**: confermati utili (rimozione peggiora MAE +0.3%). Mantenuti.
# MAGIC - Decisioni documentate in Columbus_Experiments_v1.1.ipynb.
# MAGIC
# MAGIC ### v1.1 ? 2026-04-17
# MAGIC - **Daily forecast**: aggregazione giornaliera al posto di settimanale; orizzonte 28 giorni
# MAGIC - **n_lags=0**: rimosso AR settimanale; direct forecasting via trend + stagionalit? + holidays
# MAGIC - **weekly_seasonality=4**: abilitata; apprende il profilo DOW dai dati
# MAGIC - **yearly_seasonality=10**: aumentato da 8
# MAGIC - **Weekend + holiday masking**: NaN nel training, nessun ffill/bfill
# MAGIC - **Sorgenti dati semplificate**: 3 file ? 2 (Historical_Columbus_2023_2025.csv + CSV 2026)
# MAGIC - **Post-processing**: yhat1 diretto (get_latest_forecast rimosso)
# MAGIC - **Bias correction**: aggiunta colonna Forecast_BC
# MAGIC - **Section 15 eliminata**: disaggregazione non pi? necessaria
# MAGIC
# MAGIC ### v1.0 ? baseline
# MAGIC - 8 serie incluse (Frame ? Weekend Shift e OptiSource escluse: no dati sufficienti)
# MAGIC - Aggregazione settimanale, n_lags=8, orizzonte 52 settimane
# MAGIC - Disaggregazione DOW con scale factors statici (Section 15)
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed

# Modulo condiviso (repo root)
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common import kelly_common as kc

print('Libraries loaded.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuration

# COMMAND ----------

BASE_PATH = f"{kc.input_volume_base('col')}/input"

# Sorgenti dati
FILE_HISTORICAL = f'{BASE_PATH}/Historical_Columbus_2023_2025.csv'
FILE_CURRENT    = f'{BASE_PATH}/Dallas & Columbus DC Absenteeism 2026.csv'

# Cost center -> Dept mapping
CC_DEPT = {
    'E715L035CO': 'Lens',
    'E715O891CO': 'Frame',
    'E715O057CO': 'Frame',
    'E715V035CO': 'OptiSource',  # escluso: solo 18 obs
    'E715O219CO': 'Lens',
    'E715O369CO': 'ECP (Contacts)',
    'E715O035CO': 'Lens',
}

# Shift selezionati
SHIFT_MAP = {
    '1': '1st Shift',
    'K': '2nd Shift',
    'G': 'Weekend Shift',
}

# Serie ESCLUSE dall'analisi di fattibilita'
EXCLUDED_SERIES = {
    ('Frame', 'G'),        # nessun dato disponibile
    ('OptiSource', '1'),   # solo 18 osservazioni (dal 2026-03-16)
}

# Pipeline parameters
LOCATION      = 'Columbus DC'
START_DATE    = pd.Timestamp('2023-05-01')
FORECAST_DAYS = 28    # orizzonte forecast: 4 settimane calendario (~20 giorni lavorativi)

SEED = 42
set_random_seed(SEED)

print('Configuration OK.')
print(f'Serie escluse: {EXCLUDED_SERIES}')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Data

# COMMAND ----------

# --- Historical file (2023-2025) ---
df_hist = pd.read_csv(FILE_HISTORICAL)
df_hist['Date'] = pd.to_datetime(df_hist['Date'], errors='coerce')
for col in ['AbsHours', 'ProdHours', 'TotalHours']:
    df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')
df_hist['Shift'] = df_hist['Shift'].astype(str).str.strip()

found_cc = set(df_hist['CostCenter'].dropna().unique()) & set(CC_DEPT)
print(f'CostCenter trovati in df_hist : {sorted(found_cc)}')
if len(found_cc) < len(CC_DEPT):
    print(f'  WARNING mancanti: {sorted(set(CC_DEPT) - found_cc)}')

# --- Current file (Jan 2026 - present) ---
df26 = pd.read_csv(FILE_CURRENT)
df26 = df26.rename(columns={'Home_Org_Unit': 'OrgUnit'})
df26['Date'] = pd.to_datetime(df26['Date'], errors='coerce')
for col in ['AbsHours', 'ProdHours', 'TotalHours']:
    df26[col] = pd.to_numeric(df26[col], errors='coerce')
if 'Shift' in df26.columns:
    df26['Shift'] = df26['Shift'].astype(str).str.strip()

print(f'df_hist : {df_hist["Date"].min().date()} -> {df_hist["Date"].max().date()}  ({len(df_hist):,} righe)')
print(f'df26    : {df26["Date"].min().date()} -> {df26["Date"].max().date()}  ({len(df26):,} righe)')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Clean & Build Dataset

# COMMAND ----------

COLS = ['Work_Location', 'CostCenter', 'Shift', 'Date', 'AbsHours', 'ProdHours', 'TotalHours']

# df26 in ultima posizione -> vince sull'overlap Gen 2026
# v1.4 FIX: senza drop_duplicates le righe sovrapposte venivano sommate due volte
# dal groupby successivo (stessa logica gia' presente in Dallas).
raw = (
    pd.concat([df_hist[COLS], df26[COLS]], ignore_index=True)
    .drop_duplicates(subset=['CostCenter', 'Shift', 'Date'], keep='last')
    .sort_values('Date')
    .reset_index(drop=True)
)

# Filtra Columbus DC
raw = raw[raw['Work_Location'] == LOCATION].copy()

# Filtra CostCenter e Shift target
raw = raw[raw['CostCenter'].isin(CC_DEPT)].copy()
raw = raw[raw['Shift'].isin(SHIFT_MAP)].copy()

# Mapping Dept e Shift_Label
raw['Dept']        = raw['CostCenter'].map(CC_DEPT)
raw['Shift_Label'] = raw['Shift'].map(SHIFT_MAP)

# Escludi serie non fattibili
raw = raw[~raw.apply(lambda r: (r['Dept'], r['Shift']) in EXCLUDED_SERIES, axis=1)].copy()

# ID serie = Dept + Shift_Label
raw['ID'] = raw['Dept'] + ' — ' + raw['Shift_Label']

# Aggregazione GIORNALIERA per ID
daily = (
    raw
    .groupby(['Date', 'ID'], as_index=False)
    .agg(AbsHours=('AbsHours', 'sum'), TotalHours=('TotalHours', 'sum'))
)

# y = AbsHours / TotalHours; NaN se TotalHours == 0
daily['y'] = (daily['AbsHours'] / daily['TotalHours'].replace(0, np.nan)).clip(lower=0, upper=1.0)
daily = daily.rename(columns={'Date': 'ds'})
daily = daily[daily['ds'] >= START_DATE].reset_index(drop=True)

max_date = daily['ds'].max()
min_date = daily['ds'].min()

print(f'Dataset: {min_date.date()} -> {max_date.date()}')
print(f'Serie  : {sorted(daily["ID"].unique())}')
print(f'Giorni per serie (righe fonte):')
print(daily.groupby('ID')['ds'].count().to_string())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Complete Time Series (fill missing weeks)

# COMMAND ----------

import holidays as hol

# Range calendario completo: ogni giorno da min_date a max_date
full_days  = pd.date_range(start=min_date, end=max_date, freq='D')
full_index = pd.MultiIndex.from_product(
    [full_days, daily['ID'].unique()],
    names=['ds', 'ID']
)

df = (
    pd.DataFrame(index=full_index)
    .reset_index()
    .merge(daily[['ds', 'ID', 'y']], on=['ds', 'ID'], how='left')
)

# Festività US + Christmas Eve
years         = df['ds'].dt.year.unique().tolist()
us_hol        = hol.US(years=years)
holiday_dates = pd.to_datetime(list(us_hol.keys()))
xmas_eve      = pd.to_datetime([f'{y}-12-24' for y in years])
all_holidays  = pd.DatetimeIndex(holiday_dates).union(pd.DatetimeIndex(xmas_eve))

# Weekend (Sab=5, Dom=6) e festività -> NaN
# NeuralProphet ignora i NaN nella loss: non si impara nulla da questi giorni
is_weekend = df['ds'].dt.dayofweek >= 5
is_holiday = df['ds'].isin(all_holidays)
df.loc[is_weekend | is_holiday, 'y'] = np.nan

# NESSUN ffill/bfill: i giorni mancanti rimangono NaN
df = df[['ds', 'ID', 'y']].sort_values(['ds', 'ID']).reset_index(drop=True)

total_days   = df.shape[0]
working_days = int((~is_weekend & ~is_holiday).sum() / daily['ID'].nunique())
print(f'Dataset completo : {df.shape}')
print(f'Giorni calendario per serie : {len(full_days)}')
print(f'Giorni lavorativi stimati   : {working_days}')
print(f'y non-NaN (dati effettivi)  : {df["y"].notna().sum():,}')
print(f'y NaN (weekend/holiday/gap) : {df["y"].isna().sum():,}')
print(f'\nNaN per serie:')
print(df.groupby('ID')['y'].apply(lambda x: x.isna().sum()).to_string())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Events — US Holidays & Custom Events

# COMMAND ----------

custom_events = {
    'School_Start_OH': [
        '2023-08-22', '2024-08-20', '2025-08-19', '2026-08-18',
    ],
    'School_End_OH': [
        '2023-05-25', '2024-05-23', '2025-05-22', '2026-05-21',
    ],
    'Super_Bowl': [
        '2023-02-12', '2024-02-11', '2025-02-09', '2026-02-08',
    ],
    # Ramadan: primo giorno (inizio digiuno) e Eid al-Fitr (fine)
    # v1.2: incluso dopo esperimenti (MAE -1.8%, consistente su 5/8 serie)
    'Ramadan_Start': [
        '2023-03-22', '2024-03-10', '2025-03-01', '2026-02-18',
    ],
    'Ramadan_End': [
        '2023-04-21', '2024-04-09', '2025-03-30', '2026-03-19',
    ],
    # Eid al-Adha (Festa del Sacrificio): seconda festivita' islamica maggiore.
    # Date da avvistamento lunare (~stima); 2026 cade a fine maggio.
    'Eid_al_Adha': [
        '2023-06-28', '2024-06-16', '2025-06-06', '2026-05-27',
    ],
}

df_events_wide = kc.events_dict_to_wide(custom_events)

df = df.merge(df_events_wide, on='ds', how='left')
event_cols = list(custom_events.keys())
df[event_cols] = df[event_cols].fillna(0).astype(int)

print('Events merged.')
print(df[event_cols].sum().to_string())
print('\nDate eventi nel dataset:')
print(df[df[event_cols].any(axis=1)][['ds', 'ID'] + event_cols]
      .drop_duplicates('ds').to_string(index=False))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train / Test Split

# COMMAND ----------

# train_df     : tutti i dati disponibili (modello di produzione)
# train_vintage: addestrato FORECAST_DAYS giorni fa per validazione

split_date         = max_date
split_date_vintage = max_date - pd.Timedelta(days=FORECAST_DAYS)

train_df         = df.copy()
train_df_vintage = df[df['ds'] <= split_date_vintage].copy()
test_df          = df[df['ds'] > split_date_vintage].copy()

print(f'split_date         : {split_date.date()}')
print(f'split_date_vintage : {split_date_vintage.date()}')
print(f'train_df         : {train_df["ds"].min().date()} -> {train_df["ds"].max().date()}  ({len(train_df):,} righe)')
print(f'train_df_vintage : {train_df_vintage["ds"].min().date()} -> {train_df_vintage["ds"].max().date()}  ({len(train_df_vintage):,} righe)')
print(f'test_df          : {test_df["ds"].min().date()} -> {test_df["ds"].max().date()}  ({len(test_df):,} righe)')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. NeuralProphet — Model Definition

# COMMAND ----------

def build_model():
    m = NeuralProphet(
        # Modello giornaliero -- no AR (n_lags=0): trend + stagionalita' + holidays + eventi
        # Parametri ottimizzati via HPT Bayesiano (Columbus_HPT_v1.0, Trial #97, 100 trial)
        n_lags              = 0,
        n_forecasts         = FORECAST_DAYS,
        # Ottimizzatore
        learning_rate       = 0.00190,   # v1.3: 0.003 -> 0.00190 (import. 0.293)
        loss_func           = 'MAE',     # v1.3: Huber -> MAE (import. 0.255)
        # Trend
        n_changepoints      = 12,        # v1.3: 5 -> 12 (trend piu flessibile)
        trend_reg           = 0.071,     # v1.3: 0.3 -> 0.071
        trend_global_local  = 'local',
        # Stagionalita'
        seasonality_mode    = 'additive',
        yearly_seasonality  = 6,         # v1.3: 10 -> 6 (import. 0.280 -- era overfitting)
        weekly_seasonality  = 4,         # invariato (import. 0.047)
        daily_seasonality   = False,
        seasonality_reg     = 0.159,     # v1.3: 0.3 -> 0.159
        season_global_local = 'local',
        epochs              = 99,        # v1.3: fisso per riproducibilita
        # v1.4: intervallo di previsione 90% (pinball loss aggiuntiva sui quantili)
        quantiles           = kc.QUANTILES,
    )
    # Festivita US federali con finestra +/-1 giorno
    m = m.add_country_holidays('US', lower_window=-1, upper_window=1)
    # Custom events Ohio con finestra +/-1 giorno
    m.add_events(list(custom_events.keys()), lower_window=-1, upper_window=1)
    m.set_plotting_backend('plotly')
    return m

print('Model builder defined.')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Fit — Current Model

# COMMAND ----------

COLS_DROP = ['AbsHours', 'TotalHours']
train_df         = train_df.drop(columns=COLS_DROP, errors='ignore').reset_index(drop=True)
train_df_vintage = train_df_vintage.drop(columns=COLS_DROP, errors='ignore').reset_index(drop=True)

m = build_model()
metrics = m.fit(train_df, freq='D')
print(metrics.tail())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Fit — Vintage Model (per validazione)

# COMMAND ----------

# m_vintage = build_model()
# metrics_vintage = m_vintage.fit(train_df_vintage, freq='D')
# print(metrics_vintage.tail())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Forecast

# COMMAND ----------

# Future events: occorrenze degli eventi custom nel periodo di forecast.
forecast_end         = max_date + pd.Timedelta(days=FORECAST_DAYS)
vintage_forecast_end = split_date_vintage + pd.Timedelta(days=FORECAST_DAYS)

future_events_long         = kc.build_future_events_long(custom_events, max_date, forecast_end)
future_events_vintage_long = kc.build_future_events_long(custom_events, split_date_vintage, vintage_forecast_end)

print(f'Forecast corrente  : {max_date.date()} -> {forecast_end.date()}')
n1 = len(future_events_long) if future_events_long is not None else 0
print(f'  Future events    : {n1} occorrenze')

future = m.make_future_dataframe(
    train_df, periods=FORECAST_DAYS, events_df=future_events_long
)
# future_vintage = m_vintage.make_future_dataframe(
#     train_df_vintage, periods=FORECAST_DAYS, events_df=future_events_vintage_long
# )

forecast         = m.predict(future)
# forecast_vintage = m_vintage.predict(future_vintage)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Post-processing

# COMMAND ----------

# Con n_lags=0: yhat1 per ogni riga e' la prediction diretta per quella data.
# v1.4: estrae anche i quantili 5%/95% -> Forecast_Lower / Forecast_Upper
df_forecast         = kc.extract_direct_forecast(forecast, col_name='Forecast')
# df_forecast_vintage = kc.extract_direct_forecast(forecast_vintage, col_name='Forecast_Vintage', with_bounds=False)

print('Forecast post-processing done.')
print(df_forecast[['ds', 'ID', 'Forecast', 'Forecast_Lower', 'Forecast_Upper']].tail(10).to_string(index=False))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Final Merge

# COMMAND ----------

merged_df = (
    df
    .merge(df_forecast[['ds', 'ID', 'Forecast', 'Forecast_Lower', 'Forecast_Upper']], on=['ds', 'ID'], how='outer')
    .rename(columns={'y': 'Actual'})
    .sort_values(['ds', 'ID'])
    .reset_index(drop=True)
)
for _c in kc.VINTAGE_COLS:
    merged_df[_c] = np.nan                       # placeholder: popolate dalla cella carry-forward
merged_df = merged_df[kc.STANDARD_COLS]

# Maschera weekend + festività nei forecast (giorni non lavorativi -> NaN)
is_non_working = (merged_df['ds'].dt.dayofweek >= 5) | merged_df['ds'].isin(all_holidays)
for col in kc.FORECAST_COLS:
    merged_df.loc[is_non_working, col] = np.nan
merged_df = kc.mask_bounds_like_point(merged_df)

print(f'merged_df: {merged_df.shape}')
print(f'Giorni non lavorativi mascherati: {is_non_working.sum():,}')
merged_df.tail(10)


# COMMAND ----------

# # -- Last Year Naive Benchmark -- Bias Comparison -------------------------
# # Bias = mean(Predicted - Actual)
# #   > 0  -> sovrastima (overestimate)
# #   < 0  -> sottostima (underestimate)
# #   ~ 0  -> modello non sistematicamente distorto
# #
# # Confronto: Forecast_Vintage  vs  Last Year (-52 settimane)
# # nel validation window (ultime FORECAST_DAYS giornate lavorative)

# SHIFT = pd.Timedelta(weeks=52)

# eval_win = merged_df[
#     (merged_df['ds'] > split_date_vintage) &
#     (merged_df['ds'] <= split_date)
# ].copy()

# ly_lookup = (
#     merged_df[['ds', 'ID', 'Actual']]
#     .rename(columns={'ds': 'ds_ly', 'Actual': 'Last_Year'})
#     .assign(ds=lambda x: x['ds_ly'] + SHIFT)
#     .drop(columns='ds_ly')
# )
# eval_win = eval_win.merge(ly_lookup, on=['ds', 'ID'], how='left')

# is_nw = (eval_win['ds'].dt.dayofweek >= 5) | eval_win['ds'].isin(all_holidays)
# eval_win.loc[is_nw, 'Last_Year'] = np.nan

# def bias_direction(b):
#     if pd.isna(b):   return 'n/a'
#     if b >  0.005:   return 'sovrastima'
#     if b < -0.005:   return 'sottostima '
#     return 'neutro     '

# def compute_bias(actual, pred):
#     mask = actual.notna() & pred.notna()
#     a = actual[mask].to_numpy(float)
#     p = pred[mask].to_numpy(float)
#     if len(a) == 0:
#         return np.nan, 0
#     return round(np.mean(p - a), 4), len(a)

# # -- Bias globale ----------------------------------------------------------
# bias_model, n_model = compute_bias(eval_win['Actual'], eval_win['Forecast_Vintage'])
# bias_ly,    n_ly    = compute_bias(eval_win['Actual'], eval_win['Last_Year'])
# delta_bias          = round(bias_model - bias_ly, 4)

# print(f"=== Bias Comparison -- validation window {split_date_vintage.date()} -> {split_date.date()} ===\n")
# print(f"{'Modello':<22} {'Bias':>8}  {'Direzione':<14}  {'n':>5}")
# print("-" * 58)
# print(f"{'Forecast Vintage':<22} {bias_model:>8.4f}  {bias_direction(bias_model):<14}  {n_model:>5}")
# print(f"{'Last Year (-52w)':<22} {bias_ly:>8.4f}  {bias_direction(bias_ly):<14}  {n_ly:>5}")
# print("-" * 58)
# print(f"{'Delta Bias (M - LY)':<22} {delta_bias:>8.4f}")
# print()
# if abs(delta_bias) < 0.002:
#     print("  -> Bias simile tra modello e Last Year")
# elif delta_bias < 0:
#     print(f"  -> Modello meno distorto di Last Year di {abs(delta_bias):.4f} h/gg")
# else:
#     print(f"  -> Last Year meno distorto di {abs(delta_bias):.4f} h/gg rispetto al modello")

# # -- Bias per area/serie ---------------------------------------------------
# rows = []
# for uid in sorted(eval_win['ID'].unique()):
#     sub = eval_win[eval_win['ID'] == uid]
#     bm, _ = compute_bias(sub['Actual'], sub['Forecast_Vintage'])
#     bl, _ = compute_bias(sub['Actual'], sub['Last_Year'])
#     db    = round(bm - bl, 4) if not (pd.isna(bm) or pd.isna(bl)) else np.nan
#     rows.append({'ID': uid, 'Bias_Model': bm, 'Dir_Model': bias_direction(bm),
#                  'Bias_LY': bl, 'Dir_LY': bias_direction(bl), 'Delta_Bias': db})

# bench_detail = pd.DataFrame(rows).set_index('ID')
# print(f"\n=== Bias per area/serie ===")
# print(f"{'ID':<30} {'Bias_Model':>10}  {'Dir_Model':<14}  {'Bias_LY':>8}  {'Dir_LY':<14}  {'Delta_Bias':>10}")
# print("-" * 98)
# for idx, r in bench_detail.iterrows():
#     print(f"{str(idx):<30} {r['Bias_Model']:>10.4f}  {r['Dir_Model']:<14}  {r['Bias_LY']:>8.4f}  {r['Dir_LY']:<14}  {r['Delta_Bias']:>10.4f}")
# print()
# print("(Delta_Bias < 0 -> modello meno distorto; Delta_Bias > 0 -> LY meno distorto)")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Evaluation Metrics

# COMMAND ----------

# def calc_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
#     """MAE, Bias, RMSE, SMAPE -- esclude giorni con Actual o Forecast NaN."""
#     mask = actual.notna() & forecast.notna()
#     a = actual[mask].to_numpy(dtype=float)
#     f = forecast[mask].to_numpy(dtype=float)
#     if len(a) == 0:
#         return dict(n=0, MAE=np.nan, Bias=np.nan, RMSE=np.nan, SMAPE=np.nan)
#     mae   = np.mean(np.abs(f - a))
#     bias  = np.mean(f - a)
#     rmse  = np.sqrt(np.mean((f - a) ** 2))
#     den   = np.abs(a) + np.abs(f)
#     smape = np.mean(np.where(den > 0, 2 * np.abs(f - a) / den, 0.0)) * 100
#     return dict(n=len(a), MAE=round(mae,4), Bias=round(bias,4), RMSE=round(rmse,4), SMAPE=round(smape,2))


# eval_df = merged_df[
#     (merged_df['ds'] > split_date_vintage) &
#     (merged_df['ds'] <= split_date)
# ].copy()

# g = calc_metrics(eval_df['Actual'], eval_df['Forecast_Vintage'])
# print(f'=== Metriche globali -- eval window {split_date_vintage.date()} -> {split_date.date()} ===')
# for k, v in g.items():
#     print(f'  {k:>6}: {v}')

# print('\n=== Metriche per Serie ===')
# rows = []
# for uid in sorted(eval_df['ID'].unique()):
#     sub = eval_df[eval_df['ID'] == uid].reset_index(drop=True)
#     row = calc_metrics(sub['Actual'], sub['Forecast_Vintage'])
#     rows.append({'ID': uid, **row})

# id_metrics = pd.DataFrame(rows).set_index('ID')
# print(id_metrics.to_string())


# COMMAND ----------

# eval_df['error'] = pd.to_numeric(eval_df['Forecast_Vintage'], errors='coerce') - pd.to_numeric(eval_df['Actual'], errors='coerce')
# eval_df['month'] = eval_df['ds'].dt.to_period('M').astype(str)

# print('=== Bias medio per mese nel validation window ===')
# print(eval_df.groupby('month')['error'].mean().round(4).to_string())

# print('\n=== Actual medio vs Forecast_Vintage medio per mese (giorni lavorativi) ===')
# print(eval_df.groupby('month')[['Actual','Forecast_Vintage']].mean().round(4).to_string())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Visualization

# COMMAND ----------

# START_PLOT = max_date - pd.Timedelta(days=90)
# STOP_PLOT  = max_date + pd.Timedelta(days=FORECAST_DAYS)
# BG_COLOR   = '#0b1c44'

# C_ACTUAL   = '#76b3fa'
# C_VINTAGE  = '#a07dfa'
# C_FORECAST = '#f7b267'

# series  = sorted(merged_df['ID'].unique())
# N_COLS  = 2
# n_rows  = int(np.ceil(len(series) / N_COLS))

# plt.style.use('dark_background')
# fig, axes = plt.subplots(n_rows, N_COLS, figsize=(16, n_rows * 4.2))
# axes_flat  = axes.flatten()

# for i, serie in enumerate(series):
#     ax  = axes_flat[i]
#     sub = merged_df[
#         merged_df['ds'].between(START_PLOT, STOP_PLOT) &
#         (merged_df['ID'] == serie)
#     ]

#     ax.set_facecolor(BG_COLOR)
#     ax.spines[['top','right','left','bottom']].set_visible(False)

#     ax.axvspan(split_date_vintage, max_date,
#                color=C_VINTAGE, alpha=0.07, zorder=0)
#     ax.axvspan(max_date, STOP_PLOT,
#                color=C_FORECAST, alpha=0.08, zorder=0)
#     ax.axvline(max_date, color='white', linewidth=0.8, linestyle='--', alpha=0.4)

#     ax.plot(sub['ds'], sub['Actual'],
#             color=C_ACTUAL, linewidth=1.4, alpha=0.9, label='Actual')
#     ax.plot(sub['ds'], sub['Forecast_Vintage'],
#             color=C_VINTAGE, linewidth=1.2, linestyle='--', alpha=0.8, label='Forecast Vintage')
#     ax.plot(sub['ds'], sub['Forecast'],
#             color=C_FORECAST, linewidth=2.0, alpha=0.95, label='Forecast')

#     actuals = sub['Actual'].dropna()
#     if len(actuals) > 0:
#         p95  = float(np.nanpercentile(actuals, 95))
#         ymax = max(p95 * 1.5, 0.12)
#         ax.set_ylim(0, min(ymax, 1.0))

#     ax.set_title(serie, color='white', fontsize=9, loc='left', pad=6)
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
#     ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7.5)
#     ax.tick_params(colors='#9aaabb', labelsize=7.5)
#     ax.grid(axis='y', color='white', alpha=0.06, linewidth=0.5)

#     if i == 0:
#         handles = [
#             plt.Line2D([0],[0], color=C_ACTUAL,   linewidth=1.4, label='Actual'),
#             plt.Line2D([0],[0], color=C_VINTAGE,  linewidth=1.2, linestyle='--', label='Forecast Vintage'),
#             plt.Line2D([0],[0], color=C_FORECAST, linewidth=2.0, label='Forecast'),
#         ]
#         ax.legend(handles=handles, fontsize=7.5, loc='upper right',
#                   framealpha=0.25, facecolor='#0d1f4a', edgecolor='none', labelcolor='white')

# for ax in axes_flat[len(series):]:
#     ax.set_visible(False)

# fig.patch.set_facecolor(BG_COLOR)
# fig.suptitle('Columbus DC — Absenteeism Daily Forecast per Serie',
#              color='white', fontsize=13, y=1.01)
# plt.tight_layout()
# plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Export

# COMMAND ----------

# DBTITLE 1,Cell 35
# -- Carry-forward Forecast_Vintage da file precedente -----------------------
# Congela il Forecast del run precedente (ormai passato) dentro Forecast_Vintage,
# costruendo lo storico out-of-sample reale, e mantiene il vintage gia' accumulato.
# merged_df conserva i nuovi Forecast (futuro) e Actual.
import re

OUTPUT_TABLE = kc.forecast_table("col")
FREEZE_UNTIL = pd.Timestamp.today().normalize()   # "oggi"; usa max_date per congelare solo dove c'e' actual

def _run_date(p):
    m = re.search(r'(\d{2}-\d{2}-\d{4})', p.name)
    return pd.to_datetime(m.group(1), format='%m-%d-%Y') if m else pd.NaT

# Carica il vintage precedente dalla Delta table.
# None SOLO se la tabella non esiste (primo run); ogni altro errore viene rilanciato
# per non azzerare silenziosamente lo storico Forecast_Vintage.
prev = kc.read_delta_or_none(spark, OUTPUT_TABLE)

if prev is None or prev.empty:
    print('Nessuna tabella precedente: Forecast_Vintage lasciato invariato.')
else:
    # v1.5: congela anche i bound (Forecast_Vintage_Lower/Upper) con la stessa logica
    vintage_all, _vmeta = kc.carry_forward_vintage(prev, FREEZE_UNTIL)

    # applica al merged_df corrente (sovrascrive i placeholder)
    merged_df = (
        merged_df.drop(columns=kc.VINTAGE_COLS, errors='ignore')
                 .merge(vintage_all, on=['ds', 'ID'], how='left')
                 [kc.STANDARD_COLS]
                 .sort_values(['ds', 'ID']).reset_index(drop=True)
    )

    # maschera giorni non lavorativi anche sul vintage
    is_nw = (merged_df['ds'].dt.dayofweek >= 5) | merged_df['ds'].isin(all_holidays)
    for _c in kc.VINTAGE_COLS:
        merged_df.loc[is_nw, _c] = np.nan
    merged_df = kc.mask_bounds_like_point(
        merged_df, 'Forecast_Vintage', 'Forecast_Vintage_Lower', 'Forecast_Vintage_Upper')

    print(f'Vintage da tabella Delta')
    _lvd = _vmeta['last_vintage_date']
    print(f'  last_vintage_date = {_lvd.date() if pd.notna(_lvd) else "N/A"} | punti congelati = {_vmeta["n_frozen"]}')
    print(f'  vintage totale = {merged_df["Forecast_Vintage"].notna().sum()} punti')

# Salva il nuovo merged_df nella tabella Delta (schema standard 7 colonne, round 4)
_n_rows = kc.write_forecast_table(spark, merged_df, OUTPUT_TABLE)
print(f'Salvato su tabella Delta: {OUTPUT_TABLE} ({_n_rows} righe)')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Rolling 8-week Evaluation — v1.3 (Human baseline 7%)
# MAGIC
# MAGIC Confronto **NeuralProphet v1.3 Columbus** vs **Human baseline (7% flat)** su 8 settimane rolling (zero leakage).
# MAGIC
# MAGIC **Config:** MAE loss · n_changepoints=12 · yearly_seasonality=6 · weekly_seasonality=4 · lr=0.00190 · epochs=99
# MAGIC **Periodo:** 8 lunedi dinamici prima di `max_date` | **Metodologia:** vintage settimanale, zero leakage, forecast 28gg estratto per settimana di eval.
# MAGIC **Differenza vs Dallas:** Human flat = **7%** (Columbus) invece di 5% (Dallas).
# MAGIC

# COMMAND ----------

# import time
# import logging

# # Sopprimi output NeuralProphet durante il loop
# for _ln in ['NP', 'NP.forecaster', 'NP.config', 'NP.df_utils',
#             'NP.data', 'NP.data.processing']:
#     logging.getLogger(_ln).setLevel(logging.ERROR)

# HUMAN_RATE   = 0.07   # Human flat forecast Columbus: 7% ogni giorno lavorativo
# n_eval_weeks = 8

# # -- 8 lunedi di eval (dinamici su max_date) ------------------------------
# last_monday = max_date - pd.Timedelta(days=max_date.dayofweek)
# last_eval_monday = (last_monday - pd.Timedelta(weeks=1)
#                     if max_date.dayofweek == 0 else last_monday)
# eval_mondays = [last_eval_monday - pd.Timedelta(weeks=(n_eval_weeks - 1 - i))
#                 for i in range(n_eval_weeks)]

# print(f'Eval period : {eval_mondays[0].date()} -> {(eval_mondays[-1] + pd.Timedelta(days=4)).date()}')
# print(f'Vintages    : {[str(d.date()) for d in eval_mondays]}')
# print(f'Runtime stimato: ~{n_eval_weeks * 3}-{n_eval_weeks * 4} min\n')

# # -- Rolling loop ---------------------------------------------------------
# roll_results = []
# t0 = time.time()

# for w, monday in enumerate(eval_mondays, 1):
#     # Training: dati strettamente prima del lunedi di eval (zero leakage)
#     df_roll = df[df['ds'] < monday].copy()

#     # Giorni lavorativi della settimana di eval (lun-ven, no holiday)
#     eval_days    = [monday + pd.Timedelta(days=d) for d in range(7)]
#     eval_working = pd.DatetimeIndex(
#         [d for d in eval_days if d.dayofweek < 5 and d not in all_holidays]
#     )
#     if len(eval_working) == 0:
#         print(f'  Week {w} ({monday.date()}): nessun giorno lavorativo, skip')
#         continue

#     # Fit vintage model
#     m_roll = build_model()
#     m_roll.fit(df_roll, freq='D')

#     # Forecast 28 giorni
#     roll_ev = kc.build_future_events_long(custom_events,
#         df_roll['ds'].max(), monday + pd.Timedelta(days=FORECAST_DAYS)
#     )
#     future_roll = m_roll.make_future_dataframe(
#         df_roll, periods=FORECAST_DAYS, events_df=roll_ev
#     )
#     fc_roll = m_roll.predict(future_roll)

#     # Estrai la settimana di eval
#     fc_week = (
#         fc_roll[fc_roll['ds'].isin(eval_working)][['ds', 'ID', 'yhat1']]
#         .copy()
#         .assign(yhat1=lambda x: x['yhat1'].clip(0, 1))
#     )

#     actuals_week = (
#         df[df['ds'].isin(eval_working)][['ds', 'ID', 'y']]
#         .rename(columns={'y': 'Actual'})
#     )
#     week_df = (
#         actuals_week
#         .merge(fc_week.rename(columns={'yhat1': 'AI'}), on=['ds', 'ID'], how='left')
#         .assign(Human=HUMAN_RATE, week_monday=monday)
#     )
#     roll_results.append(week_df)

#     valid = week_df.dropna(subset=['Actual', 'AI'])
#     ai_mae_w = float(np.mean(np.abs(valid['AI'] - valid['Actual']))) if len(valid) else float('nan')
#     print(f'  Week {w}/8 ({monday.date()})  AI_MAE={ai_mae_w:.4f}  {time.time()-t0:.0f}s')

# roll_df = pd.concat(roll_results, ignore_index=True)
# print(f'\nRolling eval completato: {len(roll_df)} righe  runtime {time.time()-t0:.0f}s')


# COMMAND ----------

# # -- Metriche per area ----------------------------------------------------
# areas_eval = sorted(roll_df['ID'].unique())

# def calc_metrics(sub):
#     m = sub[sub['Actual'].notna() & sub['AI'].notna()]
#     if len(m) == 0:
#         return None
#     return (float(np.mean(np.abs(m['AI']    - m['Actual']))),
#             float(np.mean(m['AI']    - m['Actual'])),
#             float(np.mean(np.abs(m['Human'] - m['Actual']))),
#             float(np.mean(m['Human'] - m['Actual'])),
#             len(m))

# summary_rows = []
# for area in areas_eval + ['GLOBAL']:
#     sub = roll_df if area == 'GLOBAL' else roll_df[roll_df['ID'] == area]
#     res = calc_metrics(sub)
#     if res is None:
#         continue
#     ai_mae, ai_bias, hm_mae, hm_bias, n = res
#     winner = 'AI' if ai_mae < hm_mae else 'Human'
#     delta  = (hm_mae - ai_mae) / hm_mae * 100
#     summary_rows.append(dict(
#         Area=area, AI_MAE=ai_mae, Human_MAE=hm_mae,
#         AI_Bias=ai_bias, Human_Bias=hm_bias,
#         Winner=winner, AI_vs_Human_pct=round(delta, 1), n=n
#     ))

# summary_df = pd.DataFrame(summary_rows)
# print('-- Rolling 8-week Eval Columbus v1.3  (MAE loss, Human 7%) --')
# print(summary_df[['Area','AI_MAE','Human_MAE','AI_Bias','Human_Bias',
#                    'Winner','AI_vs_Human_pct']]
#       .to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# # -- Save -----------------------------------------------------------------
# EVAL_DIR = BASE_PATH / 'eval_results'
# EVAL_DIR.mkdir(exist_ok=True)
# roll_df.to_csv(   EVAL_DIR / 'eval_8w_daily_columbus_v13_h7.csv',   index=False)
# summary_df.to_csv(EVAL_DIR / 'eval_8w_summary_columbus_v13_h7.csv', index=False)
# print(f'\nFile salvati in {EVAL_DIR}')

# # -- Plot -----------------------------------------------------------------
# area_rows  = [r for r in summary_rows if r['Area'] != 'GLOBAL']
# global_row = next(r for r in summary_rows if r['Area'] == 'GLOBAL')
# areas_plot = [r['Area']       for r in area_rows]
# ai_maes    = [r['AI_MAE']     for r in area_rows]
# hm_maes    = [r['Human_MAE']  for r in area_rows]
# ai_biases  = [r['AI_Bias']    for r in area_rows]
# hm_biases  = [r['Human_Bias'] for r in area_rows]

# x_pos = np.arange(len(areas_plot))
# w_bar = 0.36
# C_AI  = '#2166ac'
# C_HUM = '#d6604d'

# fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# # Pannello 1: MAE
# ax1 = axes[0]
# b1 = ax1.bar(x_pos - w_bar/2, ai_maes, w_bar, label='AI v1.3',     color=C_AI,  alpha=0.85)
# b2 = ax1.bar(x_pos + w_bar/2, hm_maes, w_bar, label='Human (7%)',  color=C_HUM, alpha=0.85)
# for b in list(b1) + list(b2):
#     ax1.text(b.get_x()+b.get_width()/2, b.get_height()+.001,
#              f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)
# ax1.set_xticks(x_pos); ax1.set_xticklabels(areas_plot, rotation=22, ha='right', fontsize=8)
# ax1.set_ylabel('MAE'); ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)
# ax1.set_title('MAE per serie', fontweight='bold')

# # Pannello 2: Bias
# ax2 = axes[1]
# ax2.bar(x_pos - w_bar/2, ai_biases, w_bar, label='AI v1.3',    color=C_AI,  alpha=0.85)
# ax2.bar(x_pos + w_bar/2, hm_biases, w_bar, label='Human (7%)', color=C_HUM, alpha=0.85)
# ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
# ax2.set_xticks(x_pos); ax2.set_xticklabels(areas_plot, rotation=22, ha='right', fontsize=8)
# ax2.set_ylabel('Bias (Pred - Actual)'); ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
# ax2.set_title('Bias per serie', fontweight='bold')

# fig.suptitle(
#     f"Rolling 8-week Eval Columbus v1.3 (Human 7%)  |  "
#     f"AI Global MAE={global_row['AI_MAE']:.4f}   "
#     f"Human Global MAE={global_row['Human_MAE']:.4f}   "
#     f"({global_row['AI_vs_Human_pct']:+.1f}%)",
#     fontsize=11, fontweight='bold'
# )
# plt.tight_layout()
# plt.savefig(EVAL_DIR / 'eval_8w_columbus_v13_h7_plot.png', dpi=150, bbox_inches='tight')
# plt.show()
# print('Plot salvato.')


# COMMAND ----------

# # -- Heatmap winner ----------------------------------------
# import matplotlib.colors as mcolors

# n_weeks = len(weeks)
# n_areas = len(areas_t)

# fig, ax1 = plt.subplots(1, 1, figsize=(12, max(4, n_weeks * 0.55 + 2)),
#                         constrained_layout=True)

# winner_num = winner_pivot.replace({'AI': 1, 'HUM': 0}).astype(float)
# cmap_win   = mcolors.LinearSegmentedColormap.from_list('rg', ['#d7191c', '#1a9641'])
# ax1.imshow(winner_num.values, cmap=cmap_win, vmin=0, vmax=1, aspect='auto')

# for ri, wk in enumerate(winner_pivot.index):
#     for ci, area in enumerate(winner_pivot.columns):
#         val = winner_pivot.loc[wk, area]
#         txt = 'AI' if val == 'AI' else 'Human'
#         ax1.text(ci, ri, txt, ha='center', va='center', fontsize=8,
#                  fontweight='bold', color='white')

# ax1.set_xticks(range(n_areas))
# ax1.set_xticklabels(winner_pivot.columns, rotation=30, ha='right', fontsize=7.5, color='black')
# ax1.set_yticks(range(n_weeks))
# ax1.set_yticklabels(winner_pivot.index, fontsize=8, color='black')
# ax1.tick_params(colors='black', length=0)
# ax1.set_title('Winners per week and series\n(green = AI  |  red = Human 7%)',
#               fontweight='bold', fontsize=10, color='black')

# fig.patch.set_facecolor('white')

# fig.suptitle('Rolling 8-Week Columbus v1.3 (Human 7%) - Weekly AI vs. Human Scorecard',
#              fontsize=11, fontweight='bold', color='black')
# EVAL_DIR = BASE_PATH / 'eval_results'
# plt.savefig(EVAL_DIR / 'eval_8w_columbus_v13_h7_scorecard.png', dpi=150, bbox_inches='tight',
#             facecolor='white')
# plt.show()
# print('Plot salvato.')


# COMMAND ----------

# # -- Plot: Actual vs AI vs Human per serie (rolling 8 settimane) ----------
# areas_plot = sorted(roll_df['ID'].unique())
# HUMAN_RATE = 0.07

# C_ACTUAL = '#2c7bb6'
# C_AI     = '#d7191c'
# C_HUMAN  = '#a0a0a0'

# n_areas = len(areas_plot)
# n_cols  = 3
# n_rows  = int(np.ceil((n_areas + 1) / n_cols))  # +1 per pannello bias settimanale

# fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.6))
# axes_flat = axes.flatten()

# for i, area in enumerate(areas_plot):
#     ax = axes_flat[i]
#     sub = roll_df[roll_df['ID'] == area].sort_values('ds')
#     work = sub[sub['Actual'].notna() | sub['AI'].notna()]

#     ax.plot(work['ds'], work['Actual'], color=C_ACTUAL, lw=2,
#             marker='o', ms=3.5, label='Actual', zorder=4)
#     ax.plot(work['ds'], work['AI'],     color=C_AI,     lw=1.8,
#             linestyle='--', marker='s', ms=3, label='AI v1.3', zorder=3)
#     ax.axhline(HUMAN_RATE, color=C_HUMAN, lw=1.4, linestyle=':',
#                label='Human (7%)', zorder=2)

#     both = sub[sub['Actual'].notna() & sub['AI'].notna()].sort_values('ds')
#     ax.fill_between(both['ds'], both['Actual'], both['AI'],
#                     alpha=0.15, color=C_AI)

#     m = sub[sub['Actual'].notna() & sub['AI'].notna()]
#     ai_mae  = float((m['AI'] - m['Actual']).abs().mean()) if len(m) else float('nan')
#     ai_bias = float((m['AI'] - m['Actual']).mean())       if len(m) else float('nan')

#     bias_lbl   = 'sottostima' if ai_bias < -0.005 else 'sovrastima' if ai_bias > 0.005 else 'neutro'
#     bias_color = '#d7191c'    if ai_bias < -0.005 else '#1a9641'     if ai_bias > 0.005 else '#888'
#     ax.text(0.02, 0.97,
#             f'MAE  = {ai_mae:.4f}\nBias = {ai_bias:+.4f}  ({bias_lbl})',
#             transform=ax.transAxes, fontsize=7.5, va='top', family='monospace',
#             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85,
#                       ec=bias_color, lw=1.2))

#     for monday in roll_df['week_monday'].drop_duplicates().sort_values():
#         ax.axvline(monday, color='#cccccc', lw=0.6, linestyle='-', zorder=1)

#     ax.set_title(area, fontsize=9, fontweight='bold', pad=4)
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
#     ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=7)
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1%}'))
#     ax.set_ylabel('Abs Rate', fontsize=8)
#     ax.set_ylim(bottom=0)
#     ax.grid(alpha=0.2, linestyle=':')
#     if i == 0:
#         ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)

# # Pannello finale: bias settimanale per serie
# ax_b = axes_flat[n_areas]
# cmap = plt.get_cmap('tab10')
# for j, area in enumerate(areas_plot):
#     sub = roll_df[roll_df['ID'] == area].copy()
#     weekly_bias = (
#         sub[sub['Actual'].notna() & sub['AI'].notna()]
#         .groupby('week_monday')
#         .apply(lambda g: (g['AI'] - g['Actual']).mean())
#         .reset_index(name='bias')
#     )
#     ax_b.plot(weekly_bias['week_monday'], weekly_bias['bias'],
#               marker='o', ms=4, lw=1.4, label=area, color=cmap(j % 10))

# ax_b.axhline(0, color='black', lw=0.8, linestyle='--')
# ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
# ax_b.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
# plt.setp(ax_b.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=7)
# ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:+.2%}'))
# ax_b.set_ylabel('Bias (AI - Actual)', fontsize=8)
# ax_b.set_title('Bias settimanale per serie', fontsize=9, fontweight='bold', pad=4)
# ax_b.legend(fontsize=6.5, loc='lower left', framealpha=0.9, ncol=2)
# ax_b.grid(alpha=0.2, linestyle=':')

# # Spegni pannelli vuoti
# for k in range(n_areas + 1, len(axes_flat)):
#     axes_flat[k].set_visible(False)

# global_res = roll_df[roll_df['Actual'].notna() & roll_df['AI'].notna()]
# g_mae  = float((global_res['AI'] - global_res['Actual']).abs().mean())
# g_bias = float((global_res['AI'] - global_res['Actual']).mean())
# fig.suptitle(
#     f'Rolling 8-week Columbus v1.3 - Actual vs AI vs Human  |  '
#     f'Global MAE={g_mae:.4f}   Bias={g_bias:+.4f}   (Human flat = 7%)',
#     fontsize=11, fontweight='bold', y=1.005
# )

# plt.tight_layout()
# EVAL_DIR = BASE_PATH / 'eval_results'
# plt.savefig(EVAL_DIR / 'eval_8w_columbus_v13_h7_actual_vs_forecast.png', dpi=150, bbox_inches='tight')
# plt.show()
# print('Plot salvato.')
