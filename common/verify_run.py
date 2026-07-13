# Databricks notebook source
# MAGIC %md
# MAGIC # Verifica post-run — Delta table forecast
# MAGIC Da eseguire dopo ogni run di un job geografia (widget `geo`: atl / col / da / mx / it).
# MAGIC Ogni check stampa ✓/✗ — tutti ✓ = run valido.

# COMMAND ----------

import os, sys
_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common import kelly_common as kc

dbutils.widgets.dropdown("geo", "da", ["atl", "col", "da", "mx", "it"])
GEO = dbutils.widgets.get("geo")
TBL = kc.forecast_table(GEO)
OLD_TBL = f"`{kc.CATALOG}`.kelly.kelly_{GEO}_forecast"

print(f"Tabella verificata: {TBL}")

_failures = []

def check(name: str, bad_count: int, detail: str = ""):
    ok = bad_count == 0
    print(f"{'✓' if ok else '✗'} {name}" + (f" — {bad_count} violazioni {detail}" if not ok else ""))
    if not ok:
        _failures.append(name)

# COMMAND ----------

# DBTITLE 1,1. Schema: 9 colonne standard
cols = [f.name for f in spark.table(TBL).schema.fields]
print("Colonne:", cols)
check("schema 9 colonne standard", 0 if cols == kc.STANDARD_COLS else 1,
      f"(attese {kc.STANDARD_COLS})")

# COMMAND ----------

# DBTITLE 1,2. Coerenza quantili (current + vintage)
_q = spark.sql(f"""
    SELECT COUNT(*) AS n FROM {TBL}
    WHERE (Forecast IS NOT NULL AND (Forecast_Lower > Forecast OR Forecast_Upper < Forecast
           OR Forecast_Lower < 0 OR Forecast_Upper > 1))
       OR (Forecast_Vintage IS NOT NULL AND Forecast_Vintage_Lower IS NOT NULL
           AND (Forecast_Vintage_Lower > Forecast_Vintage OR Forecast_Vintage_Upper < Forecast_Vintage))
""").first()["n"]
check("Lower <= point <= Upper, bound in [0,1]", _q)

# COMMAND ----------

# DBTITLE 1,3. Simmetria masking (bound mai senza point)
_q = spark.sql(f"""
    SELECT COUNT(*) AS n FROM {TBL}
    WHERE (Forecast IS NULL AND (Forecast_Lower IS NOT NULL OR Forecast_Upper IS NOT NULL))
       OR (Forecast_Vintage IS NULL AND (Forecast_Vintage_Lower IS NOT NULL OR Forecast_Vintage_Upper IS NOT NULL))
""").first()["n"]
check("bound NULL dove il point e' NULL", _q)

# COMMAND ----------

# DBTITLE 1,4. Masking giorni non lavorativi (salta MX/ATL: schedule per turno)
# Spark dayofweek: 1=domenica, 7=sabato
if GEO in ("col", "da", "it"):
    _q = spark.sql(f"""
        SELECT COUNT(*) AS n FROM {TBL}
        WHERE dayofweek(ds) IN (1, 7) AND Forecast IS NOT NULL
    """).first()["n"]
    check("weekend mascherati (Forecast NULL sab/dom)", _q)
else:
    print(f"– skip (GEO={GEO}: giorni off per turno, mascherati in-notebook)")

# COMMAND ----------

# DBTITLE 1,5. Vintage: storico preservato vs vecchio schema
try:
    _old = spark.sql(f"SELECT COUNT(*) AS n FROM {OLD_TBL} WHERE Forecast_Vintage IS NOT NULL").first()["n"]
    _new = spark.sql(f"SELECT COUNT(*) AS n FROM {TBL} WHERE Forecast_Vintage IS NOT NULL").first()["n"]
    print(f"vintage vecchio schema: {_old:,} | nuovo schema: {_new:,}")
    check("vintage non perso (nuovo >= vecchio)", 0 if _new >= _old else 1)
except Exception as e:
    print(f"– skip confronto vecchio schema ({e})")

# COMMAND ----------

# DBTITLE 1,6. Ampiezza intervallo per ID (sniff test: ~0.02-0.15)
display(spark.sql(f"""
    SELECT ID,
           ROUND(AVG(Forecast_Upper - Forecast_Lower), 4) AS ampiezza_media_PI,
           COUNT(*) AS n_punti
    FROM {TBL}
    WHERE Forecast IS NOT NULL
    GROUP BY ID ORDER BY ID
"""))
_q = spark.sql(f"""
    SELECT COUNT(*) AS n FROM (
        SELECT ID, AVG(Forecast_Upper - Forecast_Lower) AS w
        FROM {TBL} WHERE Forecast IS NOT NULL GROUP BY ID
    ) WHERE w < 0.001 OR w > 0.8
""").first()["n"]
check("ampiezza PI plausibile (no ~0, no ~1)", _q, "(ID con intervallo degenere)")

# COMMAND ----------

# DBTITLE 1,7. Copertura temporale e orizzonte
_r = spark.sql(f"""
    SELECT MIN(ds) AS ds_min, MAX(ds) AS ds_max,
           MAX(CASE WHEN Actual   IS NOT NULL THEN ds END) AS last_actual,
           MAX(CASE WHEN Forecast IS NOT NULL THEN ds END) AS last_forecast
    FROM {TBL}
""").first()
print(f"range: {_r['ds_min']} → {_r['ds_max']}")
print(f"ultimo Actual: {_r['last_actual']} | ultimo Forecast: {_r['last_forecast']}")
_horizon_days = (_r["last_forecast"] - _r["last_actual"]).days if _r["last_forecast"] and _r["last_actual"] else 0
print(f"orizzonte oltre l'ultimo Actual: {_horizon_days} giorni")
check("forecast si estende oltre l'ultimo Actual", 0 if _horizon_days > 0 else 1)

# COMMAND ----------

# DBTITLE 1,8. Esito
print("=" * 60)
if _failures:
    raise RuntimeError(f"✗ VERIFICA FALLITA — check non superati: {_failures}")
print(f"✓✓ TUTTI I CHECK SUPERATI — {TBL} valida")
