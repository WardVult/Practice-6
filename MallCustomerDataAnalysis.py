import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Частина 1: Застосування t-SNE
# Завантаження даних
data = pd.read_csv('Mall_Customers.csv')

# Вибір числових колонок
numerical_data = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Стандартизація даних
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Застосування t-SNE
tsne_model = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
tsne_results = tsne_model.fit_transform(scaled_data)

# Візуалізація t-SNE результатів
plt.figure(figsize=(8, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=50, alpha=0.7, cmap='viridis')
plt.title('t-SNE visualization (Perplexity=30, Learning rate=200)')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.show()

# Зміна perplexity та learning rate
parameters = [(5, 100), (30, 200), (50, 300)]

for perplexity, lr in parameters:
    tsne_model = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr, random_state=42)
    tsne_results = tsne_model.fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=50, alpha=0.7, cmap='viridis')
    plt.title(f't-SNE visualization (Perplexity={perplexity}, Learning rate={lr})')
    plt.xlabel('t-SNE1')
    plt.ylabel('t-SNE2')
    plt.show()



#Частина 2: Порівняння PCA та t-SNE
# PCA з двома компонентами
pca_model = PCA(n_components=2)
pca_results = pca_model.fit_transform(scaled_data)

# Візуалізація PCA результатів
plt.figure(figsize=(8, 6))
plt.scatter(pca_results[:, 0], pca_results[:, 1], s=50, alpha=0.7, cmap='viridis')
plt.title('PCA visualization')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()



#Частина 3: Кластеризація на зменшених даних
# Кластеризація після PCA
optimal_k = 5  # Виберіть оптимальну кількість кластерів
kmeans_pca = KMeans(n_clusters=optimal_k, random_state=42)
data['PCA_Cluster'] = kmeans_pca.fit_predict(pca_results)

# Візуалізація кластерів PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], hue=data['PCA_Cluster'], palette='tab10', s=100)
plt.title('Clusters after PCA')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()

# Кластеризація після t-SNE
kmeans_tsne = KMeans(n_clusters=optimal_k, random_state=42)
data['tSNE_Cluster'] = kmeans_tsne.fit_predict(tsne_results)

# Візуалізація кластерів t-SNE
plt.figure(figsize=(8, 6))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=data['tSNE_Cluster'], palette='tab10', s=100)
plt.title('Clusters after t-SNE')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.show()



# Частина 4: Інтерпретація результатів
# Аналіз результатів кластеризації
original_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Original_Cluster'] = original_kmeans.fit_predict(scaled_data)

# Порівняння Silhouette Score
print('Silhouette Score on original data:', silhouette_score(scaled_data, data['Original_Cluster']))
print('Silhouette Score after PCA:', silhouette_score(pca_results, data['PCA_Cluster']))
print('Silhouette Score after t-SNE:', silhouette_score(tsne_results, data['tSNE_Cluster']))

# Середні значення характеристик для кожної групи
pca_clusters_summary = data.groupby('PCA_Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
tsne_clusters_summary = data.groupby('tSNE_Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()

print("PCA Clusters Summary:")
print(pca_clusters_summary)

print("\nt-SNE Clusters Summary:")
print(tsne_clusters_summary)
