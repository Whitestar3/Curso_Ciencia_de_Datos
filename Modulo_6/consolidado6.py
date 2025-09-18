# ================================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset
try:
    df = pd.read_csv("cambio_climatico_agricultura.csv", encoding="utf-8")
except:
    # Si no funciona, prueba con latin1
    df = pd.read_csv("cambio_climatico_agricultura.csv", encoding="latin1")

# Normalizar los nombres de columnas (quitar tildes y espacios raros)
df.columns = (
    df.columns.str.strip()
              .str.replace(" ", "_")
              .str.replace("Ã³", "o")
              .str.replace("Ã­", "i")
              .str.replace("Ã", "i")
)

print("Primeras filas del dataset:")
print(df.head())
print("\nInformación general:")
print(df.info())
print("\nEstadísticas descriptivas:")
print(df.describe())

# Análisis gráfico
sns.pairplot(df)
plt.show()

# Detectar outliers
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.title("Detección de outliers")
plt.show()

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("ó","o").str.replace("í","i")
print(df.columns)

# ================================================
# 2. PREPROCESAMIENTO Y ESCALAMIENTO
# ================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Variables predictoras (X) y variable objetivo (y)
X = df.drop(columns=["Produccion_alimentos", "Pais"])
y = df["Produccion_alimentos"]

# Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ================================================
# 3. MODELOS DE REGRESIÓN
# ================================================
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluar_modelo_regresion(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

modelos_regresion = {
    "Regresión Lineal": LinearRegression(),
    "Árbol de Decisión": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

print("\nResultados regresión:")
for nombre, modelo in modelos_regresion.items():
    mae, mse, r2 = evaluar_modelo_regresion(modelo, X_train, y_train, X_test, y_test)
    print(f"{nombre} -> MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

# ================================================
# 4. MODELOS DE CLASIFICACIÓN
# ================================================
# Crear variable categórica (impacto bajo, medio, alto)
q1, q2 = df["Produccion_alimentos"].quantile([0.33, 0.66])
def clasificar(x):
    if x <= q1:
        return "Bajo"
    elif x <= q2:
        return "Medio"
    else:
        return "Alto"

df["Impacto"] = df["Produccion_alimentos"].apply(clasificar)

# Separar para clasificación (eliminamos Produccion_alimentos, Impacto y Pais)
X_class = df.drop(columns=["Produccion_alimentos", "Impacto", "Pais"])
y_class = df["Impacto"]

# Escalar y dividir
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    scaler.fit_transform(X_class), y_class, test_size=0.2, random_state=42
)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

clasificadores = {
    "KNN": KNeighborsClassifier(),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

print("\nResultados clasificación:")
for nombre, clf in clasificadores.items():
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)
    print(f"\nModelo: {nombre}")
    print(confusion_matrix(y_test_c, y_pred_c))
    print(classification_report(y_test_c, y_pred_c))

# ================================================
# 5. OPTIMIZACIÓN DE MODELOS
# ================================================
from sklearn.model_selection import GridSearchCV

# Ejemplo: optimizar Random Forest
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid,
                       cv=5, scoring="r2")
grid_rf.fit(X_train, y_train)

print("\nMejores parámetros Random Forest:", grid_rf.best_params_)
print("Mejor R2 validación cruzada:", grid_rf.best_score_)

# ================================================
# 6. ANÁLISIS Y CONCLUSIONES
# ================================================
print("""
Conclusiones:
- La regresión lineal da una primera aproximación, pero modelos no lineales como Random Forest
  suelen mejorar la predicción.
- Para clasificación, los árboles de decisión permiten interpretar fácilmente las reglas,
  mientras que SVM y KNN pueden mejorar métricas según los datos.
- La optimización de hiperparámetros es crucial para obtener el mejor rendimiento.
""")
