import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


# need to load the clustered dataset
df = pd.read_csv('datasets/clustered_data.csv')

# count the members in clusters 

print('\n\nValues of cluster counts for k=3')
print(df['cluster_k3'].value_counts())
print('\n\nValues of cluster counts for k=5')
print(df['cluster_k5'].value_counts())


# create dendograms for the sub clusters

# for k=3

features = ["total_sessions", "username_entropy", "password_entropy", "command_entropy"]

target_cluster = 1
df_sub = df[df['cluster_k3'] == target_cluster].dropna()

print(f"Subsetting cluster k=3 == {target_cluster}: {len(df_sub)} samples") 

# perform hierarchical clustering on subset cluster 1 of k=3
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sub[features])

linkage_array = linkage(X_scaled, method='ward')

# plot the dendrogram for k=3 cluster 1
plt.figure(figsize=(10, 7))
dendrogram(linkage_array, leaf_rotation=90, leaf_font_size=8, p=30, truncate_mode='lastp')
plt.title(f'Dendrogram for Sub-cluster k=3 == {target_cluster}')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig(f'plots/sub_clustering/k3/dendrogram_k3_cluster_{target_cluster}.png', dpi=300)
plt.close()


# for k=5
target_cluster = 1
df_sub = df[df['cluster_k5'] == target_cluster].dropna()

print(f"Subsetting cluster k=5 == {target_cluster}: {len(df_sub)} samples")

# perform hierarchical clustering on subset cluster 1 of k=5
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sub[features])

linkage_array = linkage(X_scaled, method='ward')

# plot the dendrogram for k=5 cluster 1
plt.figure(figsize=(10, 7))
dendrogram(linkage_array, leaf_rotation=90, leaf_font_size=8, p=30, truncate_mode='lastp')
plt.title(f'Dendrogram for Sub-cluster k=5 == {target_cluster}')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig(f'plots/sub_clustering/k5/dendrogram_k5_cluster_{target_cluster}.png', dpi=300)
plt.close()

# silhiouette and davies-bouldin scores for sub-clusters
# k=3 cluster 1

df_sub_k3 = df[df['cluster_k3'] == target_cluster].dropna()
X_scaled_k3 = scaler.fit_transform(df_sub_k3[features])
linkage_array_k3 = linkage(X_scaled_k3, method='ward')
silhouette_avgs_k3 = []
davies_bouldin_avgs_k3 = []

for k in range(2, 21):
    cluster_labels = fcluster(linkage_array_k3, t=k, criterion='maxclust')
    
    silhouette_avg = silhouette_score(X_scaled_k3, cluster_labels)
    davies_bouldin_avg = davies_bouldin_score(X_scaled_k3, cluster_labels)

    silhouette_avgs_k3.append(silhouette_avg)
    davies_bouldin_avgs_k3.append(davies_bouldin_avg)
    
    print(f'k=3 Sub-cluster {target_cluster}, n_clusters = {k}, Silhouette Score = {silhouette_avg:.4f}, Davies-Bouldin Score = {davies_bouldin_avg:.4f}')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 21), silhouette_avgs_k3, marker='o')
plt.title('Silhouette Score vs Number of Clusters (k=3 Sub-cluster 1)')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(range(2, 21), davies_bouldin_avgs_k3, marker='o', color='orange')
plt.title('Davies-Bouldin Score vs Number of Clusters (k=3 Sub-cluster 1)')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Score')
plt.grid()
plt.tight_layout()
plt.savefig('plots/sub_clustering/k3/cluster_evaluation_scores_k3_cluster_1.png', dpi=300, bbox_inches='tight')
plt.close()

# k=5 cluster 1
df_sub_k5 = df[df['cluster_k5'] == target_cluster].dropna()
X_scaled_k5 = scaler.fit_transform(df_sub_k5[features])
linkage_array_k5 = linkage(X_scaled_k5, method='ward')
silhouette_avgs_k5 = []
davies_bouldin_avgs_k5 = []

for k in range(2, 21):
    cluster_labels = fcluster(linkage_array_k5, t=k, criterion='maxclust')
    
    silhouette_avg = silhouette_score(X_scaled_k5, cluster_labels)
    davies_bouldin_avg = davies_bouldin_score(X_scaled_k5, cluster_labels)

    silhouette_avgs_k5.append(silhouette_avg)
    davies_bouldin_avgs_k5.append(davies_bouldin_avg)
    
    print(f'k=5 Sub-cluster {target_cluster}, n_clusters = {k}, Silhouette Score = {silhouette_avg:.4f}, Davies-Bouldin Score = {davies_bouldin_avg:.4f}')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 21), silhouette_avgs_k5, marker='o')
plt.title('Silhouette Score vs Number of Clusters (k=5 Sub-cluster 1)')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(range(2, 21), davies_bouldin_avgs_k5, marker='o', color='orange')
plt.title('Davies-Bouldin Score vs Number of Clusters (k=5 Sub-cluster 1)')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Score')
plt.grid()
plt.tight_layout()
plt.savefig('plots/sub_clustering/k5/cluster_evaluation_scores_k5_cluster_1.png', dpi=300, bbox_inches='tight')
plt.close()


