import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# ================================
# 1. Datos del experimento
# ================================
grupo_A = np.array([85, 90, 78, 88, 92, 80, 86, 89, 84, 87, 91, 82, 83, 85, 88])
grupo_B = np.array([70, 72, 75, 78, 80, 68, 74, 76, 79, 77, 73, 71, 75, 78, 80])

# ================================
# 2. Estadísticas descriptivas
# ================================
media_A = np.mean(grupo_A)
media_B = np.mean(grupo_B)
std_A = np.std(grupo_A, ddof=1)  # ddof=1 → muestra
std_B = np.std(grupo_B, ddof=1)

print("Media Grupo A (Tutoría):", media_A)
print("Desviación estándar Grupo A:", std_A)
print("Media Grupo B (Control):", media_B)
print("Desviación estándar Grupo B:", std_B)

# ================================
# Representación gráfica
# ================================
plt.figure(figsize=(12,5))

# Histogramas
plt.subplot(1,2,1)
plt.hist(grupo_A, bins=6, alpha=0.7, label="Grupo A (Tutoría)", color="blue")
plt.hist(grupo_B, bins=6, alpha=0.7, label="Grupo B (Control)", color="orange")
plt.xlabel("Calificaciones")
plt.ylabel("Frecuencia")
plt.title("Distribución de notas")
plt.legend()

# Diagramas de caja
plt.subplot(1,2,2)
plt.boxplot([grupo_A, grupo_B], labels=["Grupo A (Tutoría)", "Grupo B (Control)"])
plt.ylabel("Calificaciones")
plt.title("Diagrama de caja comparativo")

plt.tight_layout()
plt.show()

# ================================
# 3. Prueba de hipótesis (t-test)
# ================================
# H0: No hay diferencia en medias
# H1: El grupo A tiene mayor rendimiento que el grupo B
t_stat, p_value = stats.ttest_ind(grupo_A, grupo_B, equal_var=False)  # Welch t-test

print("\nEstadístico t:", t_stat)
print("Valor-p (bilateral):", p_value)

# Como la hipótesis es direccional (A > B), se divide el p-valor entre 2
p_value_one_tailed = p_value / 2

if p_value_one_tailed < 0.05 and t_stat > 0:
    print("Se rechaza H0: El grupo con tutoría tiene mejor rendimiento.")
else:
    print("No se rechaza H0: No hay evidencia suficiente de diferencia.")

# ================================
# 4. Intervalo de confianza del 95%
# ================================
mean_diff = media_A - media_B

# Error estándar de la diferencia
se = np.sqrt(std_A**2/len(grupo_A) + std_B**2/len(grupo_B))

# Grados de libertad aproximados (Welch-Satterthwaite)
df = ((std_A**2/len(grupo_A) + std_B**2/len(grupo_B))**2) / \
     ((std_A**2/len(grupo_A))**2/(len(grupo_A)-1) + (std_B**2/len(grupo_B))**2/(len(grupo_B)-1))

# Intervalo de confianza
t_crit = stats.t.ppf(0.975, df)
ci_lower = mean_diff - t_crit * se
ci_upper = mean_diff + t_crit * se

print("\nDiferencia de medias (A - B):", mean_diff)
print(f"IC 95% de la diferencia: ({ci_lower:.2f}, {ci_upper:.2f})")
