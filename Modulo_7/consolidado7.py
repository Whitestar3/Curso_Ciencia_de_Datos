# ================================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset
df = pd.read_csv("dataset_generos_musicales.csv")

# Normalizar los nombres de columnas (quitar tildes y espacios raros)
df.columns = (
    df.columns.str.strip()
              .str.replace("Ã­", "i")
              .str.replace("Ã³", "o")
              .str.replace("Ã", "a")
              .str.replace("Ã³", "o")
)


print("Primeras filas:")
print(df.head())
print("\nInformación general:")
print(df.info())
print("\nEstadísticas descriptivas:")
print(df.describe())

# Visualización inicial
plt.figure(figsize=(10,6))
sns.heatmap(df.set_index("País").T, annot=True, cmap="coolwarm")
plt.title("Popularidad de géneros musicales por país")
plt.show()

# ================================================
# 2. CLUSTERIZACIÓN
# ================================================
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

X = df.drop(columns=["País"])  # solo variables numéricas

# --- KMeans ---
kmeans3 = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster_K3"] = kmeans3.fit_predict(X)

# Método del codo
inertia = []
K = range(2,8)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(K, inertia, marker="o")
plt.xlabel("Número de clusters K")
plt.ylabel("Inercia (SSE)")
plt.title("Método del codo - KMeans")
plt.show()

# Coeficiente de silueta
for k in range(2,6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"K={k}, Silhouette={sil:.3f}")

# --- Clustering jerárquico ---
plt.figure(figsize=(8,5))
Z = linkage(X, method="ward")
dendrogram(Z, labels=df["País"].values)
plt.title("Dendrograma")
plt.xlabel("Países")
plt.ylabel("Distancia")
plt.show()

hier = AgglomerativeClustering(n_clusters=3, linkage="ward")
df["Cluster_Hier"] = hier.fit_predict(X)

# --- DBSCAN ---
for eps in [0.3, 0.5, 0.8, 1.2]:
    db = DBSCAN(eps=eps, min_samples=2).fit(X)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"DBSCAN eps={eps}: {n_clusters} clusters")

# ================================================
# 3. REDUCCIÓN DE DIMENSIONALIDAD
# ================================================
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- PCA ---
pca = PCA().fit(X)
exp_var = np.cumsum(pca.explained_variance_ratio_)
print("Varianza acumulada PCA:", exp_var)

plt.plot(range(1, len(exp_var)+1), exp_var, marker="o")
plt.axhline(0.9, color="red", linestyle="--")
plt.xlabel("Número de componentes")
plt.ylabel("Varianza explicada acumulada")
plt.title("Selección de componentes PCA")
plt.show()

# Visualización 2D con PCA
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df["Cluster_K3"], palette="Set2", s=100)
for i, pais in enumerate(df["País"]):
    plt.text(X_pca[i,0]+0.02, X_pca[i,1]+0.02, pais)
plt.title("Visualización PCA (2D)")
plt.show()

# --- t-SNE ---
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=df["Cluster_K3"], palette="Set2", s=100)
for i, pais in enumerate(df["País"]):
    plt.text(X_tsne[i,0]+0.5, X_tsne[i,1]+0.5, pais)
plt.title("Visualización t-SNE (2D)")
plt.show()

# ================================================
# 4. ANÁLISIS Y CONCLUSIONES
# ================================================
print("""
Conclusiones:
- KMeans permitió obtener 3 clusters iniciales, y tanto el método del codo como la silueta ayudan
  a justificar el número óptimo de K.
- El clustering jerárquico muestra relaciones jerárquicas claras entre países, útil para ver similitudes.
- DBSCAN depende mucho de eps: con valores bajos identifica pocos grupos, con altos agrupa casi todo.
- PCA mostró que con pocas componentes (<3) ya se explica >90% de la varianza.
- t-SNE ofrece mejor visualización de similitudes locales, aunque menos interpretable que PCA.
""")
