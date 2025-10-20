import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score

df = pd.read_csv('datasets/filtered_data.csv')

linkage_array = linkage(df.values, method='average', metric='euclidean')

silhouette_avgs = []
davies_bouldin_avgs = []

for k in range(2, 21):
    cluster_labels = fcluster(linkage_array, t=k, criterion='maxclust')
    
    silhouette_avg = silhouette_score(df.values, cluster_labels)
    davies_bouldin_avg = davies_bouldin_score(df.values, cluster_labels)

    silhouette_avgs.append(silhouette_avg)
    davies_bouldin_avgs.append(davies_bouldin_avg)
    
    print(f'For n_clusters = {k}, Silhouette Score = {silhouette_avg:.4f}, Davies-Bouldin Score = {davies_bouldin_avg:.4f}')


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 21), silhouette_avgs, marker='o')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(range(2, 21), davies_bouldin_avgs, marker='o', color='orange')
plt.title('Davies-Bouldin Score vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Score')
plt.grid()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 7))
dendrogram(
    linkage_array,
    truncate_mode='lastp',
    p=30,
    show_leaf_counts=True,
    leaf_rotation=90,
)
plt.title('Dendrogram of Filtered IP Profiles')
plt.xlabel('Filtered Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

labels = fcluster(linkage_array, t=5, criterion='maxclust')
df['cluster'] = labels

print(df['cluster'].value_counts())
print(df.groupby('cluster')['total_sessions'].mean())
print(df.groupby('cluster')['username_entropy'].mean())
print(df.groupby('cluster')['password_entropy'].mean())
print(df.groupby('cluster')['command_entropy'].mean())