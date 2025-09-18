# ==============================
# Evaluación Final - Análisis de Datos de Migración
# ==============================

import pandas as pd
import numpy as np

# 1. LIMPIEZA Y TRANSFORMACIÓN DE DATOS (3 puntos)
# -------------------------------------------------

# Cargar dataset
df = pd.read_csv("migracion.csv")

# Identificar valores perdidos
print("Valores nulos por columna:")
print(df.isnull().sum())

# Reemplazar valores perdidos (ejemplo: con la media en columnas numéricas y 'Desconocido' en texto)
for col in df.select_dtypes(include=[np.number]):
    df[col] = df[col].fillna(df[col].mean())
for col in df.select_dtypes(include=[object]):
    df[col] = df[col].fillna("Desconocido")

# Detectar y filtrar outliers usando IQR en la columna "Migrantes"
Q1 = df["Cantidad_Migrantes"].quantile(0.25)
Q3 = df["Cantidad_Migrantes"].quantile(0.75)
IQR = Q3 - Q1
filtro = (df["Cantidad_Migrantes"] >= (Q1 - 1.5*IQR)) & (df["Cantidad_Migrantes"] <= (Q3 + 1.5*IQR))
df = df[filtro]

# Reemplazo en columna "Razon_Migracion"
mapeo = {"Económica": "Trabajo", "Conflicto": "Guerra", "Educación": "Estudios"}
df["Razon_Migracion"] = df["Razon_Migracion"].replace(mapeo)

# 2. ANÁLISIS EXPLORATORIO (2 puntos)
# -----------------------------------
print("\nPrimeras 5 filas:")
print(df.head())

print("\nInformación general:")
print(df.info())

print("\nDescripción estadística:")
print(df.describe())

print("\nMedia y mediana de migrantes:")
print("Media:", df["Cantidad_Migrantes"].mean())
print("Mediana:", df["Cantidad_Migrantes"].median())

print("\nPIB promedio de países de origen:", df["PIB_Origen"].mean())
print("PIB promedio de países de destino:", df["PIB_Destino"].mean())

print("\nConteo por razón de migración:")
print(df["Razon_Migracion"].value_counts())

# 3. AGRUPAMIENTO Y SUMARIZACIÓN DE DATOS (2 puntos)
# --------------------------------------------------
print("\nTotal migrantes por razón de migración:")
print(df.groupby("Razon_Migracion")["Cantidad_Migrantes"].sum())

print("\nPromedio IDH de origen por tipo de migración:")
print(df.groupby("Razon_Migracion")["IDH_Origen"].mean())

print("\nOrdenado de mayor a menor por migrantes:")
print(df.sort_values("Cantidad_Migrantes", ascending=False))

# 4. FILTROS Y SELECCIÓN DE DATOS (2 puntos)
# ------------------------------------------
print("\nMigraciones por conflicto (Guerra):")
print(df[df["Razon_Migracion"] == "Guerra"])

print("\nMigraciones con IDH destino > 0.90:")
print(df[df["IDH_Destino"] > 0.90])

# Nueva columna diferencia de IDH
df["Diferencia_IDH"] = df["IDH_Destino"] - df["IDH_Origen"]
print("\nDataFrame con Diferencia_IDH:")
print(df.head())

# 5. EXPORTACIÓN DE DATOS (1 punto)
# ---------------------------------
df.to_csv("Migracion_Limpio.csv", index=False)
print("\n✅ Archivo 'Migracion_Limpio.csv' exportado correctamente.")
