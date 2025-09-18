# ================================
# PROYECTO: Análisis de Migraciones con PySpark
# ================================

# Importar librerías
import os
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import builtins

# Forzar PySpark a usar Python correcto en Windows
os.environ["PYSPARK_PYTHON"] = r"C:\Users\julio\AppData\Local\Programs\Python\Python313\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\julio\AppData\Local\Programs\Python\Python313\python.exe"

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.stat import Correlation

# ================================
# 1. Cargar dataset
# ================================
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "migracion.csv")

# Pandas
df_pandas = pd.read_csv(csv_path)
print("✅ CSV cargado con pandas")
print(df_pandas.head())

# SparkSession
sesion_spark = SparkSession.builder.appName("AnalisisMigraciones").getOrCreate()
print("✅ SparkSession creada exitosamente")

# Spark DataFrame
df = sesion_spark.read.csv(csv_path, header=True, inferSchema=True)
print("✅ Spark DataFrame creado exitosamente")
df.show(5, truncate=False)
df.printSchema()

# ================================
# 2. Limpieza y exploración
# ================================
# Nulos
df_nulos = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
total_nulos = df_nulos.select(expr(" + ".join([f"`{c}`" for c in df_nulos.columns])).alias("total_nulos")).collect()[0][0]
print(f"Total de valores nulos: {total_nulos}")
df_nulos.show()

# Duplicados
df_duplicados = df.groupBy(df.columns).count().filter("count > 1")
if df_duplicados.count() > 0:
    print("Hay duplicados:")
    df_duplicados.show()
else:
    print("✅ No hay duplicados")

# Estadísticas descriptivas
col_num = [c[0] for c in df.dtypes if c[1] in ['int', 'bigint', 'double', 'float']]
print("📊 Estadísticas descriptivas:")
df.select(col_num).describe().show(truncate=False)

# ================================
# 3. Visualizaciones exploratorias
# ================================
df_pd = df.toPandas()

# Evolución migrantes por año
plt.figure(figsize=(8,5))
sns.lineplot(data=df_pd, x="Año", y="Cantidad_Migrantes", hue="Razon_Migracion", marker="o")
plt.title("Evolución de migraciones por año", weight="bold")
plt.tight_layout()
plt.show()

# Migrantes por razón (agrupado)
df_grouped = df_pd.groupby("Razon_Migracion", as_index=False)["Cantidad_Migrantes"].sum()

plt.figure(figsize=(7,5))
sns.barplot(data=df_grouped, x="Razon_Migracion", y="Cantidad_Migrantes")
plt.title("Migraciones totales por razón", weight="bold")
plt.tight_layout()
plt.show()

# ================================
# 4. Spark SQL
# ================================
df.createOrReplaceTempView("migraciones")
print("✅ Tabla temporal 'migraciones' creada")

consulta = """
    SELECT Razon_Migracion, COUNT(*) as total_casos,
           ROUND(AVG(PIB_Origen),2) as promedio_PIB_Origen,
           ROUND(AVG(PIB_Destino),2) as promedio_PIB_Destino
    FROM migraciones
    GROUP BY Razon_Migracion
    ORDER BY total_casos DESC
"""
resultado = sesion_spark.sql(consulta)
print("📌 Resumen por razón de migración:")
resultado.show()

# ================================
# 5. MLlib - Regresión logística
# ================================
# Target: 1 si Económica, 0 resto
df1 = df.withColumn("target", when(col("Razon_Migracion") == "Económica", 1).otherwise(0))

# Atributos numéricos
atributos = col_num.copy()
if "Año" in atributos:
    atributos.remove("Año")

assembler = VectorAssembler(inputCols=atributos, outputCol="atributos")
df_train = assembler.transform(df1).select("atributos", "target")

# Split train-test
train, test = df_train.randomSplit([0.7, 0.3], seed=42)

# Modelo
lr = LogisticRegression(featuresCol="atributos", labelCol="target")
modelo = lr.fit(train)

# Evaluación
pred_test = modelo.transform(test)
accuracy = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy").evaluate(pred_test)

print(f"🎯 Accuracy en test set: {accuracy:.2%}")
print("⚠️ Nota: El dataset es pequeño, por lo que el modelo puede estar sobreajustado.")

# ================================
# 6. Exportar resultados
# ================================
resultado_pandas = resultado.toPandas()
resultado_pandas.to_csv("resultado_final.csv", index=False)
print("✅ Archivo 'resultado_final.csv' exportado con éxito")

# ================================
# 7. Cierre
# ================================
if SparkSession.getActiveSession() is not None:
    sesion_spark.stop()
    del sesion_spark
    gc.collect()
    print("✅ SparkSession cerrada correctamente")

