import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

# Cargar y preparar datos
data = np.loadtxt("c:/repos/k27/Nuvial/anomaly_detection/Isolation_Forest/Heart - copia.csv", delimiter=",")
heart_rates = data[:, 1]  # Asumiendo que la segunda columna es el ritmo cardíaco
time = np.arange(len(heart_rates))

# Normalizar datos
scaler = StandardScaler()
heart_rates_scaled = scaler.fit_transform(heart_rates.reshape(-1, 1)).ravel()

# Detectar anomalías con Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(heart_rates.reshape(-1, 1))

# Encontrar todos los picos y valles
peaks, _ = find_peaks(heart_rates, distance=5)
valleys, _ = find_peaks(-heart_rates, distance=5)

# Filtrar solo los picos y valles que son outliers
anomaly_peaks = peaks[anomalies[peaks] == -1]
anomaly_valleys = valleys[anomalies[valleys] == -1]

# Visualización
plt.figure(figsize=(15, 8))

# Crear dos subplots: uno para el gráfico y otro para el texto
gs = plt.GridSpec(1, 2, width_ratios=[3, 1])  # Proporción 3:1 entre gráfico y texto
ax1 = plt.subplot(gs[0])

# Gráfico principal
ax1.plot(time, heart_rates, 'b-', label='Ritmo Cardíaco', alpha=0.6)

# Marcar solo los picos y valles anómalos
ax1.plot(anomaly_peaks, heart_rates[anomaly_peaks], "r^", 
         label='Picos Anómalos', markersize=10)
ax1.plot(anomaly_valleys, heart_rates[anomaly_valleys], "rv", 
         label='Valles Anómalos', markersize=10)

ax1.set_title('Detección de Anomalías en Ritmo Cardíaco', size=14)
ax1.set_ylabel('Ritmo Cardíaco')
ax1.set_xlabel('Tiempo')
ax1.legend()
ax1.grid(True)

# Crear el texto de estadísticas
stats_text = f"Estadísticas de Anomalías:\n\n"
stats_text += f"Total de muestras: {len(heart_rates)}\n\n"
stats_text += f"Picos anómalos detectados: {len(anomaly_peaks)}\n"
stats_text += f"Valles anómalos detectados: {len(anomaly_valleys)}\n\n"
stats_text += f"Valores de los picos anómalos:\n{heart_rates[anomaly_peaks].tolist()}\n\n"
stats_text += f"Valores de los valles anómalos:\n{heart_rates[anomaly_valleys].tolist()}"

# Agregar texto en el segundo subplot
ax2 = plt.subplot(gs[1])
ax2.axis('off')
ax2.text(0, 0.95, stats_text, fontsize=9, va='top', ha='left', wrap=True)

plt.tight_layout()
plt.show()
