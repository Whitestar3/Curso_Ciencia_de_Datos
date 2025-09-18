# ================================================
# PREDICCIÓN DE NATALIDAD CON REDES NEURONALES + PDF DE GRÁFICAS
# ================================================

import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------
# 1) CARGA Y EXPLORACIÓN
# -----------------------------
for enc in ["utf-8", "latin1"]:
    try:
        df = pd.read_csv("dataset_natalidad.csv", encoding=enc)
        break
    except Exception as e:
        last_err = e
else:
    raise last_err

def normaliza_cols(cols):
    s = (cols.str.strip()
              .str.replace(r"\s+", "_", regex=True)
              .str.replace("Á","A").str.replace("É","E").str.replace("Í","I").str.replace("Ó","O").str.replace("Ú","U")
              .str.replace("á","a").str.replace("é","e").str.replace("í","i").str.replace("ó","o").str.replace("ú","u")
              .str.replace("Ã¡","a").str.replace("Ã©","e").str.replace("Ã­","i").str.replace("Ã³","o").str.replace("Ãº","u")
              .str.replace("Ã±","n").str.replace("Ñ","N")
              .str.lower())
    return s

df.columns = normaliza_cols(df.columns)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cand_target = [c for c in df.columns if "natalidad" in c.lower()]
if not cand_target:
    raise ValueError("No encuentro la columna objetivo.")
target = cand_target[0]

df = df.dropna(subset=[target])
feature_cols = [c for c in num_cols if c != target]
X = df[feature_cols].values.astype("float32")
y = df[target].values.astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -----------------------------
# 2) RED NEURONAL
# -----------------------------
def build_mlp(input_dim, hidden=[64,32], activation="relu",
              dropout=0.0, l2=0.0, lr=1e-3, optimizer_name="adam"):
    model = keras.Sequential()
    reg = regularizers.l2(l2) if l2 > 0 else None

    model.add(layers.Input(shape=(input_dim,)))
    for h in hidden:
        model.add(layers.Dense(h, activation=activation, kernel_regularizer=reg))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="linear"))

    if optimizer_name.lower() == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name.lower() == "sgd":
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif optimizer_name.lower() == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=lr)
    else:
        opt = keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

early = keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True, monitor="val_loss")

configs = [
    ([64,32], "relu", 0.0, 0.0, 1e-3, "adam", 16),
    ([128,64], "relu", 0.2, 1e-4, 1e-3, "adam", 16),
]

results, histories = [], {}

for i, (hidden, act, dr, l2v, lr, opt, bs) in enumerate(configs, start=1):
    tf.keras.backend.clear_session()
    model = build_mlp(X_train.shape[1], hidden=hidden, activation=act,
                      dropout=dr, l2=l2v, lr=lr, optimizer_name=opt)

    hist = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=bs,
        callbacks=[early],
        verbose=0
    )
    histories[f"exp_{i}"] = hist.history
    y_pred = model.predict(X_test, verbose=0).ravel()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "exp": i,
        "hidden": hidden, "activation": act, "dropout": dr, "l2": l2v,
        "lr": lr, "optimizer": opt, "batch": bs,
        "test_MAE": mae, "test_RMSE": rmse, "test_R2": r2,
    })
    print(f"[Exp {i}] MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}")

res_df = pd.DataFrame(results).sort_values("test_MAE")
best_idx = int(res_df.iloc[0]["exp"])
best_hist = histories[f"exp_{best_idx}"]

# Reentrenar el mejor modelo
best_cfg = configs[best_idx-1]
model_best = build_mlp(X_train.shape[1], hidden=best_cfg[0], activation=best_cfg[1],
                       dropout=best_cfg[2], l2=best_cfg[3], lr=best_cfg[4], optimizer_name=best_cfg[5])
_ = model_best.fit(X_train, y_train, validation_split=0.2, epochs=200,
                   batch_size=best_cfg[6], callbacks=[early], verbose=0)

y_pred_best = model_best.predict(X_test, verbose=0).ravel()

# -----------------------------
# 3) IMPORTANCIA DE VARIABLES
# -----------------------------
rng = np.random.default_rng(42)
baseline_mae = mean_absolute_error(y_test, y_pred_best)
imp = []

X_test_copy = X_test.copy()
for j, col in enumerate(feature_cols):
    X_perm = X_test_copy.copy()
    X_perm[:, j] = rng.permutation(X_perm[:, j])
    y_pred_perm = model_best.predict(X_perm, verbose=0).ravel()
    mae_perm = mean_absolute_error(y_test, y_pred_perm)
    imp.append({"feature": col, "mae_increase": mae_perm - baseline_mae})

imp_df = pd.DataFrame(imp).sort_values("mae_increase", ascending=False)

# -----------------------------
# 4) GUARDAR GRÁFICAS EN PDF
# -----------------------------
with PdfPages("reporte_natalidad.pdf") as pdf:
    # Correlaciones
    plt.figure(figsize=(10,6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matriz de correlaciones")
    pdf.savefig(); plt.close()

    # Distribución de la tasa de natalidad
    plt.figure(figsize=(6,4))
    sns.histplot(df[target], kde=True)
    plt.title(f"Distribución de {target}")
    pdf.savefig(); plt.close()

    # Curvas de entrenamiento
    plt.figure(figsize=(8,4))
    plt.plot(best_hist["loss"], label="train_loss")
    plt.plot(best_hist["val_loss"], label="val_loss")
    plt.legend(); plt.title("Curvas de entrenamiento")
    pdf.savefig(); plt.close()

    # Real vs Predicho
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred_best, alpha=0.8)
    lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
    plt.plot(lims, lims, '--', linewidth=1)
    plt.xlabel("Real"); plt.ylabel("Predicho")
    plt.title("Real vs Predicho")
    pdf.savefig(); plt.close()

    # Importancia de variables
    plt.figure(figsize=(8,4))
    sns.barplot(data=imp_df, x="mae_increase", y="feature")
    plt.title("Importancia de variables")
    pdf.savefig(); plt.close()

print("✅ Reporte PDF generado: reporte_natalidad.pdf")
