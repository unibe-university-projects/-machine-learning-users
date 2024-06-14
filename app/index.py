import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Proporcionar la ruta completa del archivo CSV
file_path = 'C:/Users/USUARIO/Documents/machine-learning-users/app/social_media_interactions.csv'
df = pd.read_csv(file_path)

# Seleccionar características para el clustering
X = df[['Likes', 'Comments', 'Shares']]

# Aplicar K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Añadir los resultados de la segmentación al DataFrame original
df['Cluster'] = kmeans.labels_

# Visualización de los clusters en 2D (Likes vs Comments)
plt.figure(figsize=(10, 7))
plt.scatter(df['Likes'], df['Comments'], c=df['Cluster'], cmap='viridis')
plt.title('Segmentación de Usuarios en Redes Sociales (Likes vs Comments)')
plt.xlabel('Likes')
plt.ylabel('Comments')
plt.colorbar(label='Cluster')
plt.show()

# Visualización de los clusters en 3D (Likes vs Comments vs Shares)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df['Likes'], df['Comments'], df['Shares'], c=df['Cluster'], cmap='viridis')
ax.set_title('Segmentación de Usuarios en Redes Sociales (3D)')
ax.set_xlabel('Likes')
ax.set_ylabel('Comments')
ax.set_zlabel('Shares')
plt.colorbar(sc, label='Cluster')
plt.show()

# Mostrar los primeros 10 usuarios y su cluster asignado
print(df.head(10))

# Mostrar un resumen de los clusters
cluster_summary = df.groupby('Cluster').mean()
print("\nResumen de Clusters:\n", cluster_summary)
