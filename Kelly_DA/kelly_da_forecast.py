# Databricks notebook source
# MAGIC %md
# MAGIC # Dallas DC — Absenteeism Forecast Pipeline
# MAGIC **Version:** 1.3
# MAGIC **Freq:** Daily | **Target:** Abs_rate per Area
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
# MAGIC   carry-forward vintage, scrittura Delta standardizzata.
# MAGIC - **Fix vintage read**: `except Exception` sostituito da gestione esplicita table-not-found —
# MAGIC   un errore transiente non azzera piu silenziosamente lo storico Forecast_Vintage.
# MAGIC - **Eventi custom estesi al 2027**: l'orizzonte 28gg attraversa gennaio 2027 dai run di dicembre 2026;
# MAGIC   aggiunto Confederate_Memorial_Day 2026 (mancava).
# MAGIC
# MAGIC ### v1.3.1 — 2026-04-29 (fix Fourier collapse)
# MAGIC
# MAGIC `weekly_seasonality` ridotto da 5 a 4.
# MAGIC Con 5 Fourier terms + `season_global_local=local` il modello convergeva a S(Tuesday)=0 su tutte le aree
# MAGIC (verificato: AI=0 ogni martedi su 8/8 vintage, 5/5 aree in eval_8w_huber_v14).
# MAGIC Con 4 terms (configurazione pre-HPT) il collasso non si riproduce.
# MAGIC
# MAGIC **Rolling 8-week eval v1.3.1** (Section 14) — 2026-04-29:
# MAGIC Confronto NeuralProphet v1.3.1 vs Human (5% flat) su 8 settimane (2026-03-02 → 2026-04-25).
# MAGIC
# MAGIC | Area | AI MAE | Hum MAE | AI Bias | Hum Bias | Vincitore |
# MAGIC |------|--------|---------|---------|----------|-----------|
# MAGIC | AM - FRAMES | 0.0577 | 0.0563 | -0.0114 | -0.0163 | Human (-2.4%) |
# MAGIC | AM INVENTORY | 0.0399 | 0.0754 | -0.0251 | +0.0171 | **AI (+47.1%)** |
# MAGIC | AM-PM LENSES | 0.0503 | 0.0456 | -0.0283 | -0.0107 | Human (-10.3%) |
# MAGIC | MAINTENANCE | 0.0962 | 0.1175 | -0.0542 | -0.0300 | **AI (+18.1%)** |
# MAGIC | PM - FRAMES | 0.0386 | 0.0572 | -0.0299 | +0.0137 | **AI (+32.5%)** |
# MAGIC | **GLOBAL** | **0.0565** | **0.0704** | -0.0298 | -0.0053 | **AI (+19.7%)** |
# MAGIC
# MAGIC Rispetto alla rolling eval v1.3 (weekly_seasonality=5): MAE globale 0.0665 → 0.0565 (**-15.0%**).
# MAGIC Il fix Fourier collapse ha migliorato tutte le aree, non solo il martedi.
# MAGIC AI vince su 3/5 aree; Human vince su AM-FRAMES (-2.4%, margine minimo) e AM-PM LENSES (-10.3%).
# MAGIC Bias sistematicamente negativo (sottostima) su tutte le aree AI.
# MAGIC
# MAGIC ### v1.3 — in uso (produzione)
# MAGIC
# MAGIC > **Test v1.4 scartato — 2026-04-28**
# MAGIC > Testati due miglioramenti su 8-week rolling eval vs baseline Huber v1.3:
# MAGIC > (1) `upper_window=2` per San Jacinto, Cesar Chavez, Good Friday (post-holiday spike +2gg);
# MAGIC > (2) `is_monday` / `is_friday` come future regressors (manager: long weekend effect).
# MAGIC > Risultato: **bias globale migliorato -38%** (da -0.030 a -0.019), ma **MAE peggiorato +16%** su tutte le aree.
# MAGIC > Causa probabile: regressors Monday/Friday globali (unico coefficiente per 5 aree con profili molto diversi)
# MAGIC > introducono varianza dove l'effetto è debole (MAINTENANCE 66% zeri) e sovra-correggono altrove.
# MAGIC > **Decisione: mantenere v1.3.** Se si vuole riprendere, occorre rendere i regressors area-specifici
# MAGIC > oppure applicarli solo alle aree con effetto DOW confermato (AM-FRAMES, AM-PM LENSES).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### v1.3 — 2026-04-27
# MAGIC - **Hyperparameter tuning (Optuna TPE)**: ottimizzazione sistematica di 8 parametri NeuralProphet
# MAGIC   su 3-fold temporal CV (fold1_summer, fold2_holiday, fold3_spring).
# MAGIC   86 trial totali (36 completati, 50 pruned dal MedianPruner, 0 falliti). Documento: Dallas_HPT_v1.0.ipynb.
# MAGIC - **loss_func Huber** (override su MAE HPT-best): Huber riduce bias sistematico del 15% vs MAE loss su 8-week rolling eval.
# MAGIC   MAE loss predice la mediana condizionale (sottostima su distribuzioni right-skewed);
# MAGIC   Huber approssima la media. Singolo cambio più impattante.
# MAGIC - **epochs 80 -> 145**: il modello era sistematicamente underfit con l'auto-setting.
# MAGIC - **yearly_seasonality 10 -> 6**: meno Fourier order riduce overfitting ciclo annuale (3 anni training).
# MAGIC - **n_changepoints 5 -> 14**: più flessibilità trend; compensata da trend_reg 0.3 -> 0.455.
# MAGIC - **weekly_seasonality 4 -> 5** (poi riportato a 4 in v1.3.1 per Fourier collapse fix).
# MAGIC - **learning_rate 0.003 -> 0.00240**, **seasonality_reg 0.3 -> 0.250**.
# MAGIC - **Risultati fold3_spring** (finestra comparabile con v1.2):
# MAGIC   MAE 0.0331 -> 0.0308 (**-6.9%**); Bias -0.0138 -> -0.0145 (stabile).
# MAGIC   Per area: MAINTENANCE -29.9%, AM-PM LENSES -7.6%, PM-FRAMES -25.8%, AM-FRAMES -3.6%.
# MAGIC   AM INVENTORY +0.0040 (serie quasi-zero, dentro il rumore di stima).
# MAGIC - **CV score 3-fold**: 0.0821 -> 0.0782 (-4.8%). Fold1_summer resta difficile (MAE~0.10);
# MAGIC   gap strutturale su estate texana non affrontabile con soli hyperparameter.
# MAGIC - **HPT landscape**: convergenza netta su MAE loss, epochs 130–145, yearly_seasonality=6.
# MAGIC   n_changepoints bimodale (3–5 e 14 equivalenti); seasonality_reg/trend_reg non sensibili (plateau piatto).
# MAGIC   Score range top-10: 0.07815–0.07891 (Δ=0.00076).
# MAGIC
# MAGIC ### v1.2 — 2026-04-17
# MAGIC - **+Texas State Holidays**: aggiunte festivita statali texane come eventi custom (+-1 giorno).
# MAGIC   Miglioramento MAE globale -3.5% (0.0343 -> 0.0331); picco su MAINTENANCE -8.7%.
# MAGIC   TX ha festivita proprie (Confederate Heroes Day, TX Independence Day, San Jacinto Day,
# MAGIC   Emancipation Day TX, LBJ Day) non presenti nel calendario federale US.
# MAGIC - **Holiday window +-2**: testato, nessun miglioramento rispetto a +-1. Non incluso.
# MAGIC - **School events TX**: neutrali (rimozione +0.3%). Mantenuti per consistenza.
# MAGIC - **Ramadan**: testato, peggiora MAE +0.9%. Non incluso (composizione workforce diversa da Columbus).
# MAGIC - Decisioni documentate in Dallas_Experiments_v1.1.ipynb.
# MAGIC
# MAGIC ### v1.1 — 2026-04-17
# MAGIC - **Daily forecast**: aggregazione giornaliera al posto di settimanale; orizzonte 28 giorni
# MAGIC - **n_lags=0**: rimosso AR; direct forecasting via trend + stagionalita + holidays
# MAGIC - **weekly_seasonality=4**: abilitata; apprende il profilo DOW dai dati
# MAGIC - **yearly_seasonality=10**: aumentato da 8
# MAGIC - **Weekend + holiday masking**: NaN nel training, nessun ffill/bfill
# MAGIC - **Sorgenti dati semplificate**: 3 file -> 2 (Historical_2023_2025.csv + CSV 2026)
# MAGIC - **Post-processing**: yhat1 diretto (get_latest_forecast rimosso)
# MAGIC - **Bias correction**: aggiunta colonna Forecast_BC
# MAGIC - **Bug fix events**: corretta mappatura date eventi
# MAGIC - **Section 15 eliminata**: disaggregazione non piu necessaria
# MAGIC
# MAGIC ### v1.0 — baseline
# MAGIC - 5 aree incluse (MAINTENANCE opzionale tramite EXCLUDE_MAINTENANCE)
# MAGIC - Aggregazione settimanale, n_lags=8, orizzonte 52 settimane
# MAGIC - Disaggregazione DOW con scale factors statici (Section 15)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## TODO
# MAGIC
# MAGIC - [x] **Rolling eval v1.3.1 completata** — Section 14. Risultati nel changelog v1.3.1 e in `eval_results/eval_8w_summary_v131.csv`.

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

from pathlib import Path
BASE_PATH = Path(kc.volume_base('da')) / 'input'

FILE_HISTORICAL = BASE_PATH / 'Historical_2023_2025.csv'
FILE_CURRENT    = BASE_PATH / 'Dallas & Columbus DC Absenteeism 2026.csv'

TARGET_ORGUNITS = ['E01539', 'E01546', 'E01551', 'E02951', 'E01548', 'E02938', 'E01704', 'E01541']

AREA_MAP = {
    'E01539': 'AM-PM LENSES',
    'E01546': 'AM-PM LENSES',
    'E01551': 'MAINTENANCE',
    'E02951': 'AM INVENTORY',
    'E01548': 'AM - FRAMES',
    'E02938': 'AM - FRAMES',
    'E01704': 'AM - FRAMES',
    'E01541': 'PM - FRAMES',
}

# Imposta True per escludere MAINTENANCE (serie ~66% zeri, forecastability scarsa)
EXCLUDE_MAINTENANCE = False

START_DATE    = pd.Timestamp('2023-05-01')
LOCATION      = 'Dallas DC'
FORECAST_DAYS = 28    # orizzonte forecast: 4 settimane calendario (~20 giorni lavorativi)

SEED = 42
set_random_seed(SEED)

print('Configuration OK.')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Data

# COMMAND ----------

# --- Historical file (2023-2025) ---
df_hist = pd.read_csv(FILE_HISTORICAL)
df_hist['Date'] = pd.to_datetime(df_hist['Date'], errors='coerce')
for col in ['AbsHours', 'ProdHours', 'TotalHours']:
    df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')

found = set(df_hist['OrgUnit'].dropna().unique()) & set(TARGET_ORGUNITS)
print(f'OrgUnit trovati in df_hist : {sorted(found)}')
if len(found) < len(TARGET_ORGUNITS):
    print(f'  WARNING OrgUnit mancanti: {sorted(set(TARGET_ORGUNITS) - found)}')

# --- Current file (Jan 2026 - present) ---
df26 = pd.read_csv(FILE_CURRENT)
df26 = df26.rename(columns={'Home_Org_Unit': 'OrgUnit'})
df26['Date'] = pd.to_datetime(df26['Date'], errors='coerce')
for col in ['AbsHours', 'ProdHours', 'TotalHours']:
    df26[col] = pd.to_numeric(df26[col], errors='coerce')

print(f'df_hist : {df_hist["Date"].min().date()} -> {df_hist["Date"].max().date()}  ({len(df_hist):,} righe)')
print(f'df26    : {df26["Date"].min().date()} -> {df26["Date"].max().date()}  ({len(df26):,} righe)')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Clean & Build Dataset

# COMMAND ----------

COLS = ['Work_Location', 'OrgUnit', 'Date', 'AbsHours', 'ProdHours', 'TotalHours']

# Concat diretto dei 2 file: df26 in ultima posizione -> vince sull'overlap Gen 2026
raw = (
    pd.concat([df_hist[COLS], df26[COLS]], ignore_index=True)
    .drop_duplicates(subset=['OrgUnit', 'Date'], keep='last')
    .sort_values('Date')
    .reset_index(drop=True)
)

raw = raw[raw['Work_Location'] == LOCATION]
raw = raw[raw['OrgUnit'].isin(TARGET_ORGUNITS)]

raw['ID'] = raw['OrgUnit'].map(AREA_MAP)

if EXCLUDE_MAINTENANCE:
    raw = raw[raw['ID'] != 'MAINTENANCE']
    print('MAINTENANCE esclusa dall\'analisi.')

# Aggregazione GIORNALIERA per Area
daily = (
    raw
    .groupby(['Date', 'ID'], as_index=False)
    .agg(AbsHours=('AbsHours', 'sum'), TotalHours=('TotalHours', 'sum'))
)

# y = AbsHours / TotalHours; NaN se TotalHours == 0 (giorno senza ore registrate)
daily['y'] = (daily['AbsHours'] / daily['TotalHours'].replace(0, np.nan)).clip(lower=0, upper=1.0)
daily = daily.rename(columns={'Date': 'ds'})
daily = daily[daily['ds'] >= START_DATE].reset_index(drop=True)

max_date = daily['ds'].max()
min_date = daily['ds'].min()

print(f'Dataset: {min_date.date()} -> {max_date.date()}')
print(f'Aree    : {sorted(daily["ID"].unique())}')
print(f'Giorni per area (righe fonte):')
print(daily.groupby('ID')['ds'].count().to_string())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Complete Time Series (daily calendar — weekend & holiday masking)

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
years        = df['ds'].dt.year.unique().tolist()
us_hol       = hol.US(years=years)
holiday_dates = pd.to_datetime(list(us_hol.keys()))
xmas_eve      = pd.to_datetime([f'{y}-12-24' for y in years])
all_holidays  = pd.DatetimeIndex(holiday_dates).union(pd.DatetimeIndex(xmas_eve))

# Weekend (Sab=5, Dom=6) e festività → NaN
# NeuralProphet ignora i NaN nella loss: non si impara nulla da questi giorni
is_weekend = df['ds'].dt.dayofweek >= 5
is_holiday = df['ds'].isin(all_holidays)
df.loc[is_weekend | is_holiday, 'y'] = np.nan

# NESSUN ffill/bfill: i giorni mancanti rimangono NaN
# NeuralProphet gestisce i missing values nativamente
df = df[['ds', 'ID', 'y']].sort_values(['ds', 'ID']).reset_index(drop=True)

total_days   = df.shape[0]
working_days = int((~is_weekend & ~is_holiday).sum() / daily['ID'].nunique())
print(f'Dataset completo : {df.shape}')
print(f'Giorni calendario per area : {len(full_days)}')
print(f'Giorni lavorativi stimati   : {working_days}')
print(f'y non-NaN (dati effettivi)  : {df["y"].notna().sum():,}')
print(f'y NaN (weekend/holiday/gap) : {df["y"].isna().sum():,}')
print(f'\nNaN per area:')
print(df.groupby('ID')['y'].apply(lambda x: x.isna().sum()).to_string())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Events — US Holidays & Custom Events

# COMMAND ----------

custom_events = {
    'School_Start_TX': [
        '2023-08-14', '2024-08-12', '2025-08-11', '2026-08-10',
    ],
    'School_End_TX': [
        '2023-05-25', '2024-05-23', '2025-05-22', '2026-05-21',
    ],
    'Super_Bowl': [
        '2023-02-12', '2024-02-11', '2025-02-09', '2026-02-08', '2027-02-14',
    ],
    # Texas state holidays (non presenti nel calendario federale US)
    # v1.2: inclusi dopo esperimenti (MAE -3.5%, MAINTENANCE -8.7%)
    # v1.4: estese al 2027 (orizzonte 28gg attraversa gennaio 2027) + fix 2026 mancante
    'Confederate_Memorial_Day': ['2023-01-19', '2024-01-19', '2025-01-19', '2026-01-19', '2027-01-19'],
    'Texas_Independence_Day': ['2023-03-02', '2024-03-02', '2025-03-02', '2026-03-02', '2027-03-02'],
    'Cesar_Chavez_Day': ['2023-03-31', '2024-03-31', '2025-03-31', '2026-03-31', '2027-03-31'],
    'Good_Friday': ['2023-04-07', '2024-03-29', '2025-04-18', '2026-04-03', '2027-03-26'],
    'San_Jacinto_Day': ['2023-04-21', '2024-04-21', '2025-04-21', '2026-04-21', '2027-04-21'],
    'Lyndon_Baines_Johnson_Day': ['2023-08-27', '2024-08-27', '2025-08-27', '2026-08-27', '2027-08-27'],
    'Friday_After_Thanksgiving': ['2023-11-24', '2024-11-29', '2025-11-28', '2026-11-27', '2027-11-26'],
    'Christmas_Eve_observed': ['2023-12-22'],
    'Christmas_Eve': ['2023-12-24', '2024-12-24', '2025-12-24', '2026-12-24', '2027-12-24'],
    'Day_After_Christmas': ['2023-12-26', '2024-12-26', '2025-12-26', '2026-12-26', '2027-12-26'],
}

# Con dati giornalieri gli eventi puntano alla data esatta
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
print(f'train_df         : {train_df["ds"].min().date()} → {train_df["ds"].max().date()}  ({len(train_df):,} righe)')
print(f'train_df_vintage : {train_df_vintage["ds"].min().date()} → {train_df_vintage["ds"].max().date()}  ({len(train_df_vintage):,} righe)')
print(f'test_df          : {test_df["ds"].min().date()} → {test_df["ds"].max().date()}  ({len(test_df):,} righe)')


# COMMAND ----------

# Rimuove colonne non necessarie per NeuralProphet
COLS_DROP = ['AbsHours', 'TotalHours']

train_df         = train_df.drop(columns=COLS_DROP, errors='ignore').reset_index(drop=True)
train_df_vintage = train_df_vintage.drop(columns=COLS_DROP, errors='ignore').reset_index(drop=True)

print('Colonne train_df:', train_df.columns.tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. NeuralProphet — Model Definition

# COMMAND ----------

def build_model():
    m = NeuralProphet(
        # Modello giornaliero — no AR (n_lags=0): trend + stagionalita + holidays + eventi
        n_lags              = 0,
        n_forecasts         = FORECAST_DAYS,
        # Ottimizzatore — HPT v1.3: Huber loss (bias globale -15% vs MAE su 8w rolling eval)
        learning_rate       = 0.00240,
        loss_func           = 'Huber',
        # Trend — HPT v1.3: piu changepoints + reg piu alta per compensare
        n_changepoints      = 14,
        trend_reg           = 0.455,
        trend_global_local  = 'local',
        # Stagionalita — HPT v1.3: yearly order ridotto (meno overfit), weekly +1
        seasonality_mode    = 'additive',
        yearly_seasonality  = 6,
        weekly_seasonality  = 4,
        daily_seasonality   = False,
        seasonality_reg     = 0.250,
        season_global_local = 'local',
        # Training — HPT v1.3: epochs aumentati (era underfit con auto=80)
        epochs              = 145,
        # v1.4: intervallo di previsione 90% (pinball loss aggiuntiva sui quantili)
        quantiles           = kc.QUANTILES,
    )
    # Festivita US con finestra +/-1 giorno
    m = m.add_country_holidays('US', lower_window=-1, upper_window=1)
    # Custom events con finestra +/-1 giorno
    # Test v1.4 (2026-04-28): upper_window=2 per post-spike events + is_monday/is_friday regressors
    # scartato: bias -38% ma MAE +16% per global regressor coefficient su aree eterogenee
    m.add_events(list(custom_events.keys()), lower_window=-1, upper_window=1)
    m.set_plotting_backend('plotly')
    return m

print('Model builder defined.')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Fit — Current Model

# COMMAND ----------

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
# Con dati giornalieri le date sono esatte (non mappate alla fine settimana).

forecast_end         = max_date + pd.Timedelta(days=FORECAST_DAYS)
vintage_forecast_end = split_date_vintage + pd.Timedelta(days=FORECAST_DAYS)

future_events_long         = kc.build_future_events_long(custom_events, max_date,          forecast_end)
future_events_vintage_long = kc.build_future_events_long(custom_events, split_date_vintage, vintage_forecast_end)

print(f'Forecast corrente  : {max_date.date()} -> {forecast_end.date()}')
n1 = len(future_events_long) if future_events_long is not None else 0
print(f'  Future events    : {n1} occorrenze')
if future_events_long is not None:
    print(future_events_long.to_string(index=False))

print(f'\nForecast vintage   : {split_date_vintage.date()} -> {vintage_forecast_end.date()}')
n2 = len(future_events_vintage_long) if future_events_vintage_long is not None else 0
print(f'  Future events    : {n2} occorrenze')


# COMMAND ----------

future = m.make_future_dataframe(
    train_df,
    periods   = FORECAST_DAYS,
    events_df = future_events_long,
)
# future_vintage = m_vintage.make_future_dataframe(
#     train_df_vintage,
#     periods   = FORECAST_DAYS,
#     events_df = future_events_vintage_long,
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

# Maschera weekend + festivita' nei forecast (giorni non lavorativi -> NaN)
is_non_working = (merged_df['ds'].dt.dayofweek >= 5) | merged_df['ds'].isin(all_holidays)
for col in kc.FORECAST_COLS:
    merged_df.loc[is_non_working, col] = np.nan
merged_df = kc.mask_bounds_like_point(merged_df)

print(f'merged_df: {merged_df.shape}')
print(f'Giorni non lavorativi mascherati: {is_non_working.sum():,}')
merged_df.tail(10)


# COMMAND ----------

# # Sanity check: allineamento eval window
# print(f'Validation window : {split_date_vintage.date()} → {split_date.date()}')
# print(f'Giorni calendario : {(split_date - split_date_vintage).days}')
#
# window_df = merged_df[
#     (merged_df['ds'] > split_date_vintage) &
#     (merged_df['ds'] <= split_date)
# ]
# actual_ok   = window_df['Actual'].notna().sum()
# vintage_ok  = window_df['Forecast_Vintage'].notna().sum()
# print(f'\nRighe nel window             : {len(window_df):,}')
# print(f'Actual non-NaN (lav. con dato): {actual_ok:,}')
# print(f'Forecast_Vintage non-NaN      : {vintage_ok:,}')
#
# print('\nSample (prime 10 righe lavorative con dato):')
# print(window_df[window_df['Actual'].notna()]
#       [['ds', 'ID', 'Actual', 'Forecast_Vintage']].head(10).to_string())


# COMMAND ----------

# DBTITLE 1,Cell 31
# -- Carry-forward Forecast_Vintage (+ bounds) da Delta table ----------------
# Congela Forecast/Forecast_Lower/Forecast_Upper del run precedente (ormai
# passati) nel trio Forecast_Vintage*, mantenendo il vintage gia' accumulato.
# I bound vintage permettono di misurare la copertura empirica del PI 90%.

from pyspark.sql import functions as F

FREEZE_UNTIL = pd.Timestamp.today().normalize()   # "oggi"; usa max_date per congelare solo dove c'e' actual

# Carica il vintage precedente dalla Delta table.
# None SOLO se la tabella non esiste (primo run); ogni altro errore viene rilanciato
# per non azzerare silenziosamente lo storico Forecast_Vintage.
# Nessun select esplicito: carry_forward_vintage tollera colonne mancanti (schema vecchio).
prev_df = kc.read_delta_or_none(spark, kc.forecast_table('da'))

vintage_all, _vmeta = kc.carry_forward_vintage(prev_df, FREEZE_UNTIL)

# applica al merged_df corrente (sovrascrive i placeholder)
merged_df = (
    merged_df.drop(columns=kc.VINTAGE_COLS, errors='ignore')
             .merge(vintage_all, on=['ds', 'ID'], how='left')
             [kc.STANDARD_COLS]
             .sort_values(['ds', 'ID']).reset_index(drop=True)
)

is_nw = (merged_df['ds'].dt.dayofweek >= 5) | merged_df['ds'].isin(all_holidays)
for _c in kc.VINTAGE_COLS:
    merged_df.loc[is_nw, _c] = np.nan
merged_df = kc.mask_bounds_like_point(
    merged_df, 'Forecast_Vintage', 'Forecast_Vintage_Lower', 'Forecast_Vintage_Upper')

print(f'Vintage da: Delta table {kc.forecast_table("da")}')
_lvd = _vmeta['last_vintage_date']
print(f'  last_vintage_date = {_lvd.date() if pd.notna(_lvd) else "N/A"} | punti congelati = {_vmeta["n_frozen"]}')
print(f'  vintage totale = {merged_df["Forecast_Vintage"].notna().sum()} punti')


# COMMAND ----------

# DBTITLE 1,Cell 32
# ── Export to Delta Table ──────────────────────────────────────────────────
# Schema standard 9 colonne (kc.STANDARD_COLS: point + bounds + vintage trio),
# round(4), overwrite + overwriteSchema.
_n_rows = kc.write_forecast_table(spark, merged_df, kc.forecast_table('da'))

print(f'Export completato: {kc.forecast_table("da")} ({_n_rows} righe)')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Validation Plot — Actual vs Forecast Vintage
# MAGIC Linee giornaliere per ogni area nella finestra di validazione (`split_date_vintage → split_date`).  
# MAGIC Pannello in basso a destra: bias per area (rosso = sottostima, verde = sovrastima, grigio = neutro).
# MAGIC
# MAGIC > **Last Year Naive Benchmark** (finestra 2026-03-30 → 2026-04-27):  
# MAGIC > Forecast Vintage Bias = -0.0529 — Last Year (-52w) Bias = -0.0374.  
# MAGIC > Il modello è meno distorto di Last Year di **0.0155** punti. Per area: AM-FRAMES e PM-FRAMES il modello supera LY; AM INVENTORY, MAINTENANCE e AM-PM LENSES LY ha bias minore.

# COMMAND ----------

# # ── Dati finestra di validazione ──────────────────────────────────────────
# win = merged_df[
#     (merged_df['ds'] > split_date_vintage) &
#     (merged_df['ds'] <= split_date)
# ].copy()
#
# areas = sorted(win['ID'].unique())
#
# # ── Helper metriche ────────────────────────────────────────────────────────
# def area_metrics(sub):
#     mask = sub['Actual'].notna() & sub['Forecast_Vintage'].notna()
#     a = sub.loc[mask, 'Actual'].values
#     p = sub.loc[mask, 'Forecast_Vintage'].values
#     if len(a) == 0:
#         return np.nan, np.nan, 0
#     return float(np.mean(np.abs(p - a))), float(np.mean(p - a)), len(a)
#
# # ── Layout 2×3: 5 aree + 1 bias summary ───────────────────────────────────
# fig, axes = plt.subplots(2, 3, figsize=(17, 8))
# axes_flat = axes.flatten()
#
# COL_ACTUAL   = '#2166ac'   # blu
# COL_VINTAGE  = '#d6604d'   # arancio-rosso
# COL_FILL     = '#b2d8e8'   # azzurro chiaro fill
#
# for i, area in enumerate(areas):
#     ax = axes_flat[i]
#     sub = win[win['ID'] == area].sort_values('ds')
#     mae, bias, n = area_metrics(sub)
#
#     # Riga solo su giorni lavorativi (non-NaN)
#     work = sub[sub['Actual'].notna() | sub['Forecast_Vintage'].notna()]
#
#     ax.plot(work['ds'], work['Actual'],
#             color=COL_ACTUAL, linewidth=1.8, marker='o', markersize=3.5,
#             label='Actual', zorder=3)
#     ax.plot(work['ds'], work['Forecast_Vintage'],
#             color=COL_VINTAGE, linewidth=1.8, linestyle='--', marker='s', markersize=3.5,
#             label='Forecast Vintage', zorder=3)
#
#     # Shaded error area solo dove entrambe le serie sono presenti
#     both = sub[sub['Actual'].notna() & sub['Forecast_Vintage'].notna()].sort_values('ds')
#     ax.fill_between(both['ds'], both['Actual'], both['Forecast_Vintage'],
#                     alpha=0.18, color=COL_FILL, zorder=2)
#
#     # Annotation MAE + Bias
#     bias_color = '#d6604d' if bias < -0.005 else '#4dac26' if bias > 0.005 else '#888888'
#     bias_label = 'sottostima' if bias < -0.005 else 'sovrastima' if bias > 0.005 else 'neutro'
#     ax.text(0.02, 0.97,
#             f'MAE = {mae:.4f}\nBias = {bias:+.4f}  ({bias_label})\nn = {n}',
#             transform=ax.transAxes, fontsize=7.5, va='top', family='monospace',
#             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
#                       alpha=0.85, edgecolor=bias_color, linewidth=1.2))
#
#     ax.set_title(area, fontsize=10, fontweight='bold', pad=4)
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
#     ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))  # ogni lunedì
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=7)
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1%}'))
#     ax.set_ylabel('Abs Rate', fontsize=8)
#     ax.grid(alpha=0.25, linestyle=':')
#     ax.set_ylim(bottom=0)
#     if i == 0:
#         ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
#
# # ── Pannello 6: bias per area (bar orizzontale) ────────────────────────────
# ax6 = axes_flat[5]
#
# bias_vals, mae_vals, area_labels = [], [], []
# for area in areas:
#     sub = win[win['ID'] == area]
#     mae, bias, _ = area_metrics(sub)
#     bias_vals.append(bias)
#     mae_vals.append(mae)
#     area_labels.append(area)
#
# bar_colors = ['#d6604d' if b < -0.005 else '#4dac26' if b > 0.005 else '#aaaaaa'
#               for b in bias_vals]
#
# bars = ax6.barh(area_labels, bias_vals, color=bar_colors, alpha=0.85,
#                 edgecolor='white', height=0.55)
# ax6.axvline(0, color='black', linewidth=0.9)
#
# # Etichette bias + MAE a fianco
# for bar, bv, mv in zip(bars, bias_vals, mae_vals):
#     xoff = 0.0008 if bv >= 0 else -0.0008
#     ha   = 'left' if bv >= 0 else 'right'
#     ax6.text(bv + xoff, bar.get_y() + bar.get_height() / 2,
#              f'{bv:+.4f}  (MAE {mv:.4f})',
#              va='center', ha=ha, fontsize=7.5)
#
# ax6.set_xlabel('Bias  (Pred − Actual)', fontsize=8)
# ax6.set_title('Per-Area Bias', fontsize=10, fontweight='bold', pad=4)
# ax6.grid(axis='x', alpha=0.25, linestyle=':')
# ax6.set_xlim(
#     min(bias_vals) - 0.025,
#     max(bias_vals) + 0.025
# )
#
# # ── Titolo globale ─────────────────────────────────────────────────────────
# global_mae  = float(np.mean(mae_vals))
# global_bias = float(np.mean(bias_vals))
# fig.suptitle(
#     f'Validation Window  {split_date_vintage.date()} → {split_date.date()}  '
#     f'({FORECAST_DAYS} calendar days)  |  '
#     f'Global MAE = {global_mae:.4f}   Global Bias = {global_bias:+.4f}',
#     fontsize=11, y=1.01, fontweight='bold'
# )
#
# plt.tight_layout()
# plt.savefig(BASE_PATH / 'validation_actual_vs_vintage_v1.3.1.png', dpi=150, bbox_inches='tight')
# plt.show()
# print(f'Plot salvato: {BASE_PATH}/validation_actual_vs_vintage_v1.3.1.png')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Human vs AI — Rolling Eval (HPT reference, weekly_seasonality=5)
# MAGIC
# MAGIC > **Nota storica:** rolling eval eseguita nel notebook HPT (Sezione 8) con config `weekly_seasonality=5`
# MAGIC > (prima del fix Fourier collapse di v1.3.1). Per l'eval con config corrente v1.3.1 vedere **Section 15**.
# MAGIC
# MAGIC **Config:** Huber loss · n_changepoints=14 · yearly_seasonality=6 · **weekly_seasonality=5** · lr=0.00240 · epochs=145  
# MAGIC **Periodo:** 2026-03-02 → 2026-04-27 (8 settimane) | **Human baseline:** 5% flat
# MAGIC
# MAGIC | Area | AI MAE | Hum MAE | AI Bias | Hum Bias | Vincitore |
# MAGIC |------|--------|---------|---------|----------|-----------|
# MAGIC | AM - FRAMES | 0.0712 | 0.0563 | -0.0092 | -0.0163 | Human |
# MAGIC | AM INVENTORY | 0.0483 | 0.0754 | -0.0123 | +0.0171 | **AI** |
# MAGIC | AM-PM LENSES | 0.0532 | 0.0456 | -0.0166 | -0.0107 | Human |
# MAGIC | MAINTENANCE | 0.1200 | 0.1175 | -0.0325 | -0.0300 | Human |
# MAGIC | PM - FRAMES | 0.0397 | 0.0572 | -0.0227 | +0.0137 | **AI** |
# MAGIC | **GLOBAL** | **0.0665** | **0.0704** | -0.0187 | -0.0053 | **AI (+5.5%)** |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Rolling 8-week Evaluation — v1.3.1
# MAGIC
# MAGIC Confronto **NeuralProphet v1.3.1** vs **Human baseline (5% flat)** su 8 settimane rolling (zero leakage).
# MAGIC
# MAGIC **Config:** Huber loss · n_changepoints=14 · yearly_seasonality=6 · **weekly_seasonality=4** · lr=0.00240 · epochs=145  
# MAGIC **Periodo:** 2026-03-02 → 2026-04-25 (8 settimane) | **Metodologia:** vintage settimanale, zero leakage, forecast 28gg estratto per settimana di eval.
# MAGIC
# MAGIC ### Risultati
# MAGIC
# MAGIC | Area | AI MAE | Hum MAE | AI Bias | Hum Bias | Vincitore |
# MAGIC |------|--------|---------|---------|----------|-----------|
# MAGIC | AM - FRAMES | 0.0577 | 0.0563 | -0.0114 | -0.0163 | Human (-2.4%) |
# MAGIC | AM INVENTORY | 0.0399 | 0.0754 | -0.0251 | +0.0171 | **AI (+47.1%)** |
# MAGIC | AM-PM LENSES | 0.0503 | 0.0456 | -0.0283 | -0.0107 | Human (-10.3%) |
# MAGIC | MAINTENANCE | 0.0962 | 0.1175 | -0.0542 | -0.0300 | **AI (+18.1%)** |
# MAGIC | PM - FRAMES | 0.0386 | 0.0572 | -0.0299 | +0.0137 | **AI (+32.5%)** |
# MAGIC | **GLOBAL** | **0.0565** | **0.0704** | -0.0298 | -0.0053 | **AI (+19.7%)** |
# MAGIC
# MAGIC **AI vince: 3/5 aree + globale (+19.7%).** vs v1.3 (ws=5): MAE globale 0.0665 → 0.0565 (**-15.0%** per il fix Fourier).  
# MAGIC Bias negativo sistematico su tutte le aree AI (sottostima); Human ha bias misto.  
# MAGIC AM-FRAMES: Human vince di misura (-2.4%). AM-PM LENSES: Human più preciso (-10.3%).

# COMMAND ----------

# import time
# import logging
#
# # Sopprimi output NeuralProphet durante il loop
# for _ln in ['NP', 'NP.forecaster', 'NP.config', 'NP.df_utils',
#             'NP.data', 'NP.data.processing']:
#     logging.getLogger(_ln).setLevel(logging.ERROR)
#
# HUMAN_RATE   = 0.05   # Human flat forecast: 5% ogni giorno lavorativo
# n_eval_weeks = 8
#
# # ── 8 lunedi di eval (dinamici su max_date) ───────────────────────────────
# last_monday = max_date - pd.Timedelta(days=max_date.dayofweek)
# last_eval_monday = (last_monday - pd.Timedelta(weeks=1)
#                     if max_date.dayofweek == 0 else last_monday)
# eval_mondays = [last_eval_monday - pd.Timedelta(weeks=(n_eval_weeks - 1 - i))
#                 for i in range(n_eval_weeks)]
#
# print(f'Eval period : {eval_mondays[0].date()} -> {(eval_mondays[-1] + pd.Timedelta(days=4)).date()}')
# print(f'Vintages    : {[str(d.date()) for d in eval_mondays]}')
# print(f'Runtime stimato: ~{n_eval_weeks * 3}-{n_eval_weeks * 4} min\n')
#
# # ── Rolling loop ──────────────────────────────────────────────────────────
# roll_results = []
# t0 = time.time()
#
# for w, monday in enumerate(eval_mondays, 1):
#     # Training: dati strettamente prima del lunedi di eval (zero leakage)
#     df_roll = df[df['ds'] < monday].copy()
#
#     # Giorni lavorativi della settimana di eval (lun-ven, no holiday)
#     eval_days    = [monday + pd.Timedelta(days=d) for d in range(7)]
#     eval_working = pd.DatetimeIndex(
#         [d for d in eval_days if d.dayofweek < 5 and d not in all_holidays]
#     )
#     if len(eval_working) == 0:
#         print(f'  Week {w} ({monday.date()}): nessun giorno lavorativo, skip')
#         continue
#
#     # Fit vintage model
#     m_roll = build_model()
#     m_roll.fit(df_roll, freq='D')
#
#     # Forecast 28 giorni
#     roll_ev = kc.build_future_events_long(
#         custom_events, df_roll['ds'].max(), monday + pd.Timedelta(days=FORECAST_DAYS)
#     )
#     future_roll = m_roll.make_future_dataframe(
#         df_roll, periods=FORECAST_DAYS, events_df=roll_ev
#     )
#     fc_roll = m_roll.predict(future_roll)
#
#     # Estrai la settimana di eval
#     fc_week = (
#         fc_roll[fc_roll['ds'].isin(eval_working)][['ds', 'ID', 'yhat1']]
#         .copy()
#         .assign(yhat1=lambda x: x['yhat1'].clip(0, 1))
#     )
#
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
#
#     valid = week_df.dropna(subset=['Actual', 'AI'])
#     ai_mae_w = float(np.mean(np.abs(valid['AI'] - valid['Actual']))) if len(valid) else float('nan')
#     print(f'  Week {w}/8 ({monday.date()})  AI_MAE={ai_mae_w:.4f}  {time.time()-t0:.0f}s')
#
# roll_df = pd.concat(roll_results, ignore_index=True)
# print(f'\nRolling eval completato: {len(roll_df)} righe  runtime {time.time()-t0:.0f}s')


# COMMAND ----------

# # ── Metriche per area ─────────────────────────────────────────────────────
# areas_eval = sorted(roll_df['ID'].unique())
#
# def calc_metrics(sub):
#     m = sub[sub['Actual'].notna() & sub['AI'].notna()]
#     if len(m) == 0:
#         return None
#     return (float(np.mean(np.abs(m['AI']    - m['Actual']))),
#             float(np.mean(m['AI']    - m['Actual'])),
#             float(np.mean(np.abs(m['Human'] - m['Actual']))),
#             float(np.mean(m['Human'] - m['Actual'])),
#             len(m))
#
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
#
# summary_df = pd.DataFrame(summary_rows)
# print('── Rolling 8-week Eval v1.3.1  (Huber · weekly_seasonality=4) ──────────')
# print(summary_df[['Area','AI_MAE','Human_MAE','AI_Bias','Human_Bias',
#                    'Winner','AI_vs_Human_pct']]
#       .to_string(index=False, float_format=lambda x: f'{x:.4f}'))
#
# # ── Save ──────────────────────────────────────────────────────────────────
# EVAL_DIR = BASE_PATH / 'eval_results'
# EVAL_DIR.mkdir(exist_ok=True)
# roll_df.to_csv(   EVAL_DIR / 'eval_8w_daily_v131.csv',   index=False)
# summary_df.to_csv(EVAL_DIR / 'eval_8w_summary_v131.csv', index=False)
# print(f'\nFile salvati in {EVAL_DIR}')
#
# # ── Plot ──────────────────────────────────────────────────────────────────
# area_rows  = [r for r in summary_rows if r['Area'] != 'GLOBAL']
# global_row = next(r for r in summary_rows if r['Area'] == 'GLOBAL')
# areas_plot = [r['Area']       for r in area_rows]
# ai_maes    = [r['AI_MAE']     for r in area_rows]
# hm_maes    = [r['Human_MAE']  for r in area_rows]
# ai_biases  = [r['AI_Bias']    for r in area_rows]
# hm_biases  = [r['Human_Bias'] for r in area_rows]
#
# x_pos = np.arange(len(areas_plot))
# w_bar = 0.36
# C_AI  = '#2166ac'
# C_HUM = '#d6604d'
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# # Pannello 1: MAE
# ax1 = axes[0]
# b1 = ax1.bar(x_pos - w_bar/2, ai_maes, w_bar, label='AI v1.3.1',  color=C_AI,  alpha=0.85)
# b2 = ax1.bar(x_pos + w_bar/2, hm_maes, w_bar, label='Human (5%)', color=C_HUM, alpha=0.85)
# for b in list(b1) + list(b2):
#     ax1.text(b.get_x()+b.get_width()/2, b.get_height()+.001,
#              f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)
# ax1.set_xticks(x_pos); ax1.set_xticklabels(areas_plot, rotation=18, ha='right', fontsize=8)
# ax1.set_ylabel('MAE'); ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)
# ax1.set_title('MAE per area', fontweight='bold')
#
# # Pannello 2: Bias
# ax2 = axes[1]
# ax2.bar(x_pos - w_bar/2, ai_biases,  w_bar, label='AI v1.3.1',  color=C_AI,  alpha=0.85)
# ax2.bar(x_pos + w_bar/2, hm_biases, w_bar, label='Human (5%)', color=C_HUM, alpha=0.85)
# ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
# ax2.set_xticks(x_pos); ax2.set_xticklabels(areas_plot, rotation=18, ha='right', fontsize=8)
# ax2.set_ylabel('Bias (Pred - Actual)'); ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
# ax2.set_title('Bias per area', fontweight='bold')
#
# fig.suptitle(
#     f"Rolling 8-week Eval v1.3.1  |  "
#     f"AI Global MAE={global_row['AI_MAE']:.4f}   "
#     f"Human Global MAE={global_row['Human_MAE']:.4f}   "
#     f"({global_row['AI_vs_Human_pct']:+.1f}%)",
#     fontsize=11, fontweight='bold'
# )
# plt.tight_layout()
# plt.savefig(EVAL_DIR / 'eval_8w_v131_plot.png', dpi=150, bbox_inches='tight')
# plt.show()
# print('Plot salvato.')


# COMMAND ----------

# # ── Tabella settimanale: chi vince per area e per settimana ───────────────
# areas_t  = sorted(roll_df['ID'].unique())
# weeks    = sorted(roll_df['week_monday'].unique())
#
# # ── Calcola MAE e winner per ogni (week, area) ─────────────────────────────
# records = []
# for wk in weeks:
#     for area in areas_t:
#         sub = roll_df[(roll_df['week_monday'] == wk) & (roll_df['ID'] == area)]
#         m   = sub[sub['Actual'].notna() & sub['AI'].notna()]
#         if len(m) == 0:
#             continue
#         ai_mae  = float((m['AI']    - m['Actual']).abs().mean())
#         hm_mae  = float((m['Human'] - m['Actual']).abs().mean())
#         ai_bias = float((m['AI']    - m['Actual']).mean())
#         winner  = 'AI' if ai_mae < hm_mae else 'HUM'
#         delta   = (hm_mae - ai_mae) / hm_mae * 100
#         records.append(dict(Week=wk.strftime('%d/%m'), Area=area,
#                             AI_MAE=ai_mae, Hum_MAE=hm_mae,
#                             AI_Bias=ai_bias, Winner=winner, Delta_pct=delta))
#
# detail_df = pd.DataFrame(records)
#
# # ── Pivot: winner per (Week x Area) ───────────────────────────────────────
# winner_pivot = detail_df.pivot(index='Week', columns='Area', values='Winner')
# delta_pivot  = detail_df.pivot(index='Week', columns='Area', values='Delta_pct')
#
# print('═' * 70)
# print('  WINNERS BY WEEK AND AREA   (AI = AI wins  |  HUM = Human wins)')
# print('═' * 70)
# print(winner_pivot.to_string())
# print()
#
# # ── Conteggio vittorie per area ────────────────────────────────────────────
# ai_wins  = (winner_pivot == 'AI').sum()
# hum_wins = (winner_pivot == 'HUM').sum()
# print('── Total wins over 8 weeks ───────────────────────────────────────────')
# score_df = pd.DataFrame({'AI_wins': ai_wins, 'Human_wins': hum_wins})
# score_df['Dominant'] = score_df.apply(
#     lambda r: f"AI ({r['AI_wins']}/8)" if r['AI_wins'] > r['Human_wins']
#               else f"Human ({r['Human_wins']}/8)", axis=1)
# print(score_df.to_string())
# print()
#
# # ── Tabella completa con MAE AI e MAE Human per settimana e area ──────────
# print('── Detailed MAE by week ─────────────────────────────────────')
# for area in areas_t:
#     sub = detail_df[detail_df['Area'] == area][
#         ['Week', 'AI_MAE', 'Hum_MAE', 'AI_Bias', 'Winner', 'Delta_pct']
#     ].copy()
#     sub['Delta_pct'] = sub['Delta_pct'].map(lambda x: f'{x:+.1f}%')
#     sub['AI_MAE']    = sub['AI_MAE'].map(lambda x: f'{x:.4f}')
#     sub['Hum_MAE']   = sub['Hum_MAE'].map(lambda x: f'{x:.4f}')
#     sub['AI_Bias']   = sub['AI_Bias'].map(lambda x: f'{x:+.4f}')
#     print(f"  {area}")
#     print(sub.to_string(index=False))
#     print()
#
# # ── Heatmap winner ────────────────────────────────────────────────────
# import matplotlib.colors as mcolors
#
# n_weeks = len(weeks)
# n_areas = len(areas_t)
#
# fig, ax1 = plt.subplots(1, 1, figsize=(12, max(4, n_weeks * 0.55 + 2)),
#                         constrained_layout=True)
#
# winner_num = winner_pivot.replace({'AI': 1, 'HUM': 0}).astype(float)
# cmap_win   = mcolors.LinearSegmentedColormap.from_list('rg', ['#d7191c', '#1a9641'])
# ax1.imshow(winner_num.values, cmap=cmap_win, vmin=0, vmax=1, aspect='auto')
#
# for ri, wk in enumerate(winner_pivot.index):
#     for ci, area in enumerate(winner_pivot.columns):
#         val = winner_pivot.loc[wk, area]
#         txt = 'AI' if val == 'AI' else 'Human'
#         ax1.text(ci, ri, txt, ha='center', va='center', fontsize=8,
#                  fontweight='bold', color='white')
#
# ax1.set_xticks(range(n_areas))
# ax1.set_xticklabels(winner_pivot.columns, rotation=30, ha='right', fontsize=7.5, color='black')
# ax1.set_yticks(range(n_weeks))
# ax1.set_yticklabels(winner_pivot.index, fontsize=8, color='black')
# ax1.tick_params(colors='black', length=0)
# ax1.set_title('Winners per week and area(green = AI  |  red = Human)',fontweight='bold', fontsize=10, color='black')
#
# fig.patch.set_facecolor('white')
#
# fig.suptitle('Rolling 8-week v1.3.1 Dallas — Scorecard weekly AI vs Human',
#              fontsize=11, fontweight='bold', color='black')
# EVAL_DIR = BASE_PATH / 'eval_results'
# plt.savefig(EVAL_DIR / 'eval_8w_v131_scorecard.png', dpi=150, bbox_inches='tight',
#             facecolor='white')
# plt.show()
# print('Plot salvato.')


# COMMAND ----------

# # ── Plot: Actual vs AI vs Human per area (rolling 8 settimane) ────────────
# # roll_df disponibile dalla cella precedente; in alternativa:
# # roll_df = pd.read_csv(BASE_PATH / 'eval_results' / 'eval_8w_daily_v131.csv',
# #                       parse_dates=['ds', 'week_monday'])
#
# areas_plot = sorted(roll_df['ID'].unique())
# HUMAN_RATE = 0.05
#
# C_ACTUAL = '#2c7bb6'   # blu scuro — Actual
# C_AI     = '#d7191c'   # rosso    — AI
# C_HUMAN  = '#a0a0a0'   # grigio   — Human flat
#
# fig, axes = plt.subplots(3, 2, figsize=(16, 12))
# axes_flat = axes.flatten()
#
# for i, area in enumerate(areas_plot):
#     ax = axes_flat[i]
#     sub = roll_df[roll_df['ID'] == area].sort_values('ds')
#     work = sub[sub['Actual'].notna() | sub['AI'].notna()]
#
#     ax.plot(work['ds'], work['Actual'], color=C_ACTUAL, lw=2,
#             marker='o', ms=3.5, label='Actual', zorder=4)
#     ax.plot(work['ds'], work['AI'],     color=C_AI,     lw=1.8,
#             linestyle='--', marker='s', ms=3, label='AI v1.3.1', zorder=3)
#     ax.axhline(HUMAN_RATE, color=C_HUMAN, lw=1.4, linestyle=':',
#                label='Human (5%)', zorder=2)
#
#     # Shade errore AI vs Actual
#     both = sub[sub['Actual'].notna() & sub['AI'].notna()].sort_values('ds')
#     ax.fill_between(both['ds'], both['Actual'], both['AI'],
#                     alpha=0.15, color=C_AI)
#
#     # Metriche per area
#     m = sub[sub['Actual'].notna() & sub['AI'].notna()]
#     ai_mae  = float((m['AI'] - m['Actual']).abs().mean()) if len(m) else float('nan')
#     ai_bias = float((m['AI'] - m['Actual']).mean())       if len(m) else float('nan')
#     hm_mae  = abs(HUMAN_RATE - m['Actual'].mean())        if len(m) else float('nan')
#
#     bias_lbl   = 'sottostima' if ai_bias < -0.005 else 'sovrastima' if ai_bias > 0.005 else 'neutro'
#     bias_color = '#d7191c'    if ai_bias < -0.005 else '#1a9641'     if ai_bias > 0.005 else '#888'
#     ax.text(0.02, 0.97,
#             f'MAE  = {ai_mae:.4f}\nBias = {ai_bias:+.4f}  ({bias_lbl})',
#             transform=ax.transAxes, fontsize=7.5, va='top', family='monospace',
#             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85,
#                       ec=bias_color, lw=1.2))
#
#     # Linee verticali per separare le settimane
#     for monday in roll_df['week_monday'].drop_duplicates().sort_values():
#         ax.axvline(monday, color='#cccccc', lw=0.6, linestyle='-', zorder=1)
#
#     ax.set_title(area, fontsize=10, fontweight='bold', pad=4)
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
#     ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=7)
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1%}'))
#     ax.set_ylabel('Abs Rate', fontsize=8)
#     ax.set_ylim(bottom=0)
#     ax.grid(alpha=0.2, linestyle=':')
#     if i == 0:
#         ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
#
# # ── Pannello 6: bias settimanale per area (linee) ─────────────────────────
# ax6 = axes_flat[5]
# cmap = plt.get_cmap('tab10')
# for j, area in enumerate(areas_plot):
#     sub = roll_df[roll_df['ID'] == area].copy()
#     weekly_bias = (
#         sub[sub['Actual'].notna() & sub['AI'].notna()]
#         .groupby('week_monday')
#         .apply(lambda g: (g['AI'] - g['Actual']).mean())
#         .reset_index(name='bias')
#     )
#     ax6.plot(weekly_bias['week_monday'], weekly_bias['bias'],
#              marker='o', ms=4, lw=1.6, label=area, color=cmap(j))
#
# ax6.axhline(0, color='black', lw=0.8, linestyle='--')
# ax6.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
# ax6.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
# plt.setp(ax6.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=7)
# ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:+.2%}'))
# ax6.set_ylabel('Bias (AI - Actual)', fontsize=8)
# ax6.set_title('Bias settimanale per area', fontsize=10, fontweight='bold', pad=4)
# ax6.legend(fontsize=7, loc='lower left', framealpha=0.9)
# ax6.grid(alpha=0.2, linestyle=':')
#
# # ── Titolo globale ─────────────────────────────────────────────────────────
# global_res = roll_df[roll_df['Actual'].notna() & roll_df['AI'].notna()]
# g_mae  = float((global_res['AI'] - global_res['Actual']).abs().mean())
# g_bias = float((global_res['AI'] - global_res['Actual']).mean())
# fig.suptitle(
#     f'Rolling 8-week v1.3.1 — Actual vs AI vs Human  |  '
#     f'Global MAE={g_mae:.4f}   Bias={g_bias:+.4f}   (Human flat = 5%)',
#     fontsize=11, fontweight='bold', y=1.01
# )
#
# plt.tight_layout()
# EVAL_DIR = BASE_PATH / 'eval_results'
# plt.savefig(EVAL_DIR / 'eval_8w_v131_actual_vs_forecast.png', dpi=150, bbox_inches='tight')
# plt.show()
# print('Plot salvato.')
