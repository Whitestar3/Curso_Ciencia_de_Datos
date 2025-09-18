# ==============================
# Evaluación Final - Análisis de Atletas Olímpicos
# ==============================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 1. ANÁLISIS EXPLORATORIO DE DATOS (2 puntos)
# --------------------------------------------

# Cargar dataset
df = pd.read_csv("olimpicos.csv")

# Ver columnas
print("Columnas disponibles en el dataset:")
print(df.columns.tolist())

# Primeras filas
print("\nPrimeras 5 filas:")
print(df.head())

# Información general
print("\nInformación general:")
print(df.info())

# Estadísticas descriptivas
print("\nDescripción estadística:")
print(df.describe())

# Histograma de entrenamientos semanales
plt.figure(figsize=(8,5))
sns.histplot(df["Entrenamientos_Semanales"], bins=10, kde=True, color="blue")
plt.title("Distribución de Entrenamientos Semanales")
plt.xlabel("Entrenamientos por semana")
plt.ylabel("Frecuencia")
plt.show()

# 2. ESTADÍSTICA DESCRIPTIVA (2 puntos)
# -------------------------------------

print("\nTipos de variables por columna:")
print(df.dtypes)

print("\nMedia, mediana y moda de medallas:")
print("Media:", df["Medallas_Totales"].mean())
print("Mediana:", df["Medallas_Totales"].median())
print("Moda:", df["Medallas_Totales"].mode()[0])

print("\nDesviación estándar de la altura (cm):")
print(df["Altura_cm"].std())

# Boxplot de peso
plt.figure(figsize=(6,5))
sns.boxplot(x=df["Peso_kg"], color="orange")
plt.title("Boxplot de Peso de Atletas (kg)")
plt.show()

# 3. ANÁLISIS DE CORRELACIÓN (2 puntos)
# -------------------------------------

# Correlación de Pearson
corr = df["Entrenamientos_Semanales"].corr(df["Medallas_Totales"], method="pearson")
print("\nCorrelación de Pearson entre entrenamientos y medallas:", corr)

# Scatterplot peso vs medallas
plt.figure(figsize=(6,5))
sns.scatterplot(x="Peso_kg", y="Medallas_Totales", data=df, color="green")
plt.title("Relación entre Peso y Medallas Totales")
plt.xlabel("Peso (kg)")
plt.ylabel("Medallas Totales")
plt.show()

print("\nInterpretación: Si la correlación es cercana a 1 hay relación positiva; cercana a -1, negativa; y cercana a 0, no hay relación.")

# 4. REGRESIÓN LINEAL (2 puntos)
# -------------------------------

X = df["Entrenamientos_Semanales"]
y = df["Medallas_Totales"]

# Agregar constante
X = sm.add_constant(X)
modelo = sm.OLS(y, X).fit()

print("\nResumen del modelo de regresión lineal:")
print(modelo.summary())

print("\nCoeficientes de regresión:")
print("Intercepto:", modelo.params["const"])
print("Pendiente:", modelo.params["Entrenamientos_Semanales"])
print("\nR² del modelo:", modelo.rsquared)

# Regresión lineal
plt.figure(figsize=(6,5))
sns.regplot(x="Entrenamientos_Semanales", y="Medallas_Totales", data=df, line_kws={"color":"red"})
plt.title("Regresión Lineal: Entrenamientos vs Medallas Totales")
plt.show()

# 5. VISUALIZACIÓN DE DATOS (2 puntos)
# ------------------------------------

# Heatmap de correlación
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Correlación entre Variables Numéricas")
plt.show()

# Boxplot de medallas por deporte
plt.figure(figsize=(10,6))
sns.boxplot(x="Deporte", y="Medallas_Totales", data=df, palette="Set2")
plt.title("Distribución de Medallas por Deporte")
plt.xticks(rotation=45)
plt.show()

print("\n✅ Análisis completo realizado. Los gráficos y resultados se han mostrado en pantalla.")
