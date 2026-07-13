# Databricks notebook source
# MAGIC %md
# MAGIC # Setup one-off — schema `sbx-logistics`.`kelly-abs-forecast`
# MAGIC
# MAGIC Da eseguire **una sola volta** prima del primo run dei job nel nuovo schema:
# MAGIC 1. Crea i 5 volumi (`kelly_{atl,col,da,mx,it}_volume`).
# MAGIC 2. Copia i file dai volumi del VECCHIO schema `kelly` (input + output/logs/reports/checkpoints se presenti).
# MAGIC 3. Semina le 5 Delta table dal vecchio schema (CTAS) — lo storico `Forecast_Vintage` continua senza interruzioni.
# MAGIC
# MAGIC Idempotente: volumi/tabelle esistenti vengono saltati (`IF NOT EXISTS`), le copie sovrascrivono.
# MAGIC Il vecchio schema `kelly` NON viene toccato (solo letture).

# COMMAND ----------

import os, sys
_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common import kelly_common as kc

GEOS = ["atl", "col", "da", "mx", "it"]

OLD_SCHEMA_QUALIFIED = f"`{kc.CATALOG}`.kelly"
OLD_VOLUME_BASE = f"/Volumes/{kc.CATALOG}/kelly"

print(f"Nuovo schema : {kc.SCHEMA_QUALIFIED}")
print(f"Vecchio schema (sola lettura): {OLD_SCHEMA_QUALIFIED}")

# COMMAND ----------

# DBTITLE 1,1. Crea i volumi nel nuovo schema
for geo in GEOS:
    vol = f"kelly_{geo}_volume"
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {kc.SCHEMA_QUALIFIED}.{vol}")
    print(f"✓ volume {vol}")

# COMMAND ----------

# DBTITLE 1,2. Copia i file dai vecchi volumi
# Sottocartelle note per geografia (quelle assenti vengono saltate).
_SUBDIRS = ["input", "output", "logs", "reports", "checkpoints", "plots"]

copy_report = []
for geo in GEOS:
    for sub in _SUBDIRS:
        src = f"{OLD_VOLUME_BASE}/kelly_{geo}_volume/{sub}"
        dst = f"{kc.volume_base(geo)}/{sub}"
        try:
            n_files = len(dbutils.fs.ls(src))
        except Exception:
            continue  # sottocartella assente nel vecchio volume
        dbutils.fs.cp(src, dst, recurse=True)
        copy_report.append((geo, sub, n_files))
        print(f"✓ copiato {src} -> {dst} ({n_files} voci)")

if not copy_report:
    print("⚠ Nessuna cartella copiata — verificare che i vecchi volumi esistano.")

# COMMAND ----------

# DBTITLE 1,3. Semina le Delta table dal vecchio schema (CTAS)
# Lo schema vecchio puo' avere 5 o 7 colonne: nessun problema, la prima run
# scrive con overwriteSchema e carry_forward_vintage tollera i bound mancanti.
def _table_exists(qualified_name: str) -> bool:
    """Probe con identifier backtick-safe (catalogo/schema con trattini)."""
    try:
        spark.sql(f"SELECT 1 FROM {qualified_name} LIMIT 1")
        return True
    except Exception:
        return False

for geo in GEOS:
    src_tbl = f"{OLD_SCHEMA_QUALIFIED}.kelly_{geo}_forecast"
    dst_tbl = kc.forecast_table(geo)
    if not _table_exists(src_tbl):
        print(f"⚠ {src_tbl} non esiste — {dst_tbl} partira' senza storico vintage")
        continue
    spark.sql(f"CREATE TABLE IF NOT EXISTS {dst_tbl} AS SELECT * FROM {src_tbl}")
    print(f"✓ seminata {dst_tbl}")

# COMMAND ----------

# DBTITLE 1,4. Report finale
print("=" * 70)
print("TABELLE NEL NUOVO SCHEMA")
print("=" * 70)
for geo in GEOS:
    dst_tbl = kc.forecast_table(geo)
    try:
        _cnt = spark.table(dst_tbl).count()
        _vint = spark.table(dst_tbl).where("Forecast_Vintage IS NOT NULL").count()
        print(f"  {dst_tbl}: {_cnt:,} righe | vintage: {_vint:,}")
    except Exception as e:
        print(f"  {dst_tbl}: ASSENTE ({e})")

print()
print("=" * 70)
print("VOLUMI COPIATI")
print("=" * 70)
for geo, sub, n in copy_report:
    print(f"  kelly_{geo}_volume/{sub}: {n} voci")

print()
print("✓✓ Setup completato — eseguire common/smoke_test_kelly_common.py, poi il job DA.")
