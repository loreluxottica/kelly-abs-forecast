# Databricks notebook source
# MX03_HeadcountData_Timestamps — query definitiva per Forecast
query = """
    SELECT 
        [Numero]  AS Clerk,
        [Turno]   AS [Shift],
        [Fecha]   AS [Date],
        12        AS TotalHours,
        CASE 
            WHEN ([absenteeism] + [tardes]) > 12 THEN 12 
            ELSE ([absenteeism] + [tardes]) 
        END       AS AbsHours
    FROM [dbo].[MX03_HeadcountData_Timestamps]
    WHERE CAST([Tipo_de_Dia] AS VARCHAR(max)) = 'Hábil'
      AND [year] IN ('2025', '2026')
"""

df = spark.read \
     .format("jdbc") \
     .option("url", "jdbc:sqlserver://10.80.192.78:1433;databaseName=Business_Intelligence") \
     .option("user", dbutils.secrets.get(scope="kelly", key="jdbc_user")) \
     .option("password", dbutils.secrets.get(scope="kelly", key="jdbc_password")) \
     .option("query", query) \
     .option("encrypt", "false") \
     .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver") \
     .load()

print(f'Colonne: {df.columns}')
print(f'Rows: {df.count()}')
display(df)