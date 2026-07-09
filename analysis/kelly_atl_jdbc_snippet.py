# Databricks notebook source
query = "SELECT * FROM [Operations].[dbo].[absenteeism_by_dept_area]"

df = spark.read \
     .format("jdbc") \
     .option("url", "jdbc:sqlserver://10.80.192.78:1433;databaseName=Operations") \
     .option("user", dbutils.secrets.get(scope="kelly", key="jdbc_user")) \
     .option("password", dbutils.secrets.get(scope="kelly", key="jdbc_password")) \
     .option("query", query) \
     .option("encrypt", "false") \
     .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver") \
     .load()

# COMMAND ----------

display(df)