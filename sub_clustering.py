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
