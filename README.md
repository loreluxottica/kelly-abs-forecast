# KELLY — Absenteeism Forecast

Previsione giornaliera del tasso di assenteismo (y ∈ [0,1]) per plant/turno/reparto con NeuralProphet.
Ogni notebook di produzione è l'entry point di un job Databricks schedulato (retrain settimanale) e scrive
una Delta table con **schema standard 9 colonne**:
`ds, ID, Actual, Forecast_Vintage, Forecast, Forecast_Lower, Forecast_Upper, Forecast_Vintage_Lower, Forecast_Vintage_Upper`
(intervallo di previsione 90%, round 4). Il trio `Forecast_Vintage*` congela run dopo run ciò che la
produzione ha pubblicato (lag-1) — inclusi i bound, per misurare la copertura empirica del PI in BI.

## Layout

**Convenzione naming:** snake_case, il nome del notebook = nome della Delta table che scrive
(`kelly_<geo>_forecast.py` → `kelly.kelly_<geo>_forecast`). Nessuna versione nei nomi file:
la versione vive solo in `MODEL_VERSION` dentro il codice.

| Percorso | Contenuto |
|---|---|
| `common/kelly_common.py` | Modulo condiviso: metriche, estrazione forecast+quantili, vintage carry-forward, scrittura Delta, notifiche Teams, validazioni. Importato da tutti i notebook prod. |
| `common/smoke_test_kelly_common.py` | Smoke test del modulo — eseguirlo su Databricks dopo ogni modifica a `kelly_common.py`. |
| `Kelly_ATL/kelly_atl_forecast.py` | Atlanta DC (2 modelli A/B; vintage lag-1) → `kelly.kelly_atl_forecast` + CSV |
| `Kelly_COL/kelly_col_forecast.py` | Columbus DC → `kelly.kelly_col_forecast` |
| `Kelly_DA/kelly_da_forecast.py` | Dallas DC → `kelly.kelly_da_forecast` |
| `Kelly_MX/kelly_mx_forecast.py` | Tijuana (JDBC SQL Server) → `kelly.kelly_mx_forecast` |
| `Kelly_IT/kelly_it_forecast.py` | Sedico → `kelly.kelly_it_forecast` |
| `analysis/` | Notebook di supporto non schedulati: `kelly_atl_analysis_cv.py`, `kelly_mx_experiments.py`, snippet JDBC. |

Brasile rimosso dal repo (2026-07-09): pipeline non piu mantenuta qui; decommissionare gli eventuali job Databricks (forecast + ETL BronzeToSilver).

## Location Unity Catalog

Tabelle e volumi vivono in **`sbx-logistics`.`kelly-abs-forecast`** (catalogo/schema centralizzati in
`kelly_common.CATALOG` / `SCHEMA` — helper `kc.forecast_table(geo)` e `kc.volume_base(geo)`, unico punto
da cambiare per spostare tutto). Il vecchio schema `kelly` non viene piu scritto.

## Prerequisiti (one-off)

1. Secret scope `kelly` su Databricks con chiavi `jdbc_user`, `jdbc_password`, `teams_webhook_url`:

```
databricks secrets create-scope kelly
databricks secrets put-secret kelly jdbc_user
databricks secrets put-secret kelly jdbc_password
databricks secrets put-secret kelly teams_webhook_url
```

2. Eseguire **una volta** `common/setup_schema_kelly_abs_forecast.py` su Databricks: crea i 5 volumi nel
   nuovo schema, copia i file dai vecchi volumi e semina le 5 Delta table dal vecchio schema
   (lo storico `Forecast_Vintage` continua). Poi `common/smoke_test_kelly_common.py`, poi i job.

## Decisione: notebook, non script

I notebook restano gli entry point dei job (decisione 2026-07-09). Motivi: integrazione nativa con Databricks Jobs
(log/debug per cella), riuso del codice già risolto da `common/kelly_common.py`, nessun beneficio runtime dalla
conversione a fronte di riconfigurazione job + retest di 5 pipeline. Da rivalutare solo se diventano requisiti
CI/CD con unit test o deploy multi-workspace (in quel caso: Databricks Asset Bundles + pytest su `kelly_common`).

## Igiene repo

`lightning_logs/`, checkpoint `.ckpt`, `__pycache__/` ed export `.xlsx` sono artefatti di runtime: NON committarli
(vedi `.gitignore`). Gli output di produzione vivono su Delta table e UC Volumes, non nel repo.
