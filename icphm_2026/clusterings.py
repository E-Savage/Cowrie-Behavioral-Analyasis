import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

# 1. Load the dataset
file_path = 'datasets/ip_entropies.csv'
df = pd.read_csv(file_path)

# 2. Data Cleaning & Pre-processing
if 'total_sessions' in df.columns:
    df['total_sessions'] = pd.to_numeric(df['total_sessions'], errors='coerce')

numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(0)
original_numeric_data = df[numeric_cols]

# 3. Normalize for K-Means math
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(original_numeric_data)

# 4. Perform K-Means (k=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(normalized_values)
df['cluster_id'] = cluster_labels

# 5. Find the "Centroid IP" (The IP closest to the mathematical center)
ip_col = 'ip' if 'ip' in df.columns else df.select_dtypes(include=['object']).columns[0]
centroid_ips = []

for i in range(5):
    # Get all points in this cluster
    cluster_indices = np.where(cluster_labels == i)[0]
    cluster_points = normalized_values[cluster_indices]
    
    # Get the centroid for this cluster
    centroid = kmeans.cluster_centers_[i].reshape(1, -1)
    
    # Find the index of the point with the minimum distance to the centroid
    distances = cdist(cluster_points, centroid, 'euclidean')
    closest_point_idx = cluster_indices[np.argmin(distances)]
    
    # Get the actual IP for that point
    centroid_ips.append(df.iloc[closest_point_idx][ip_col])

# 6. Create the Final Centroid Summary
original_centroids = df.groupby('cluster_id')[numeric_cols].mean().reset_index()
original_centroids['representative_ip'] = centroid_ips

# 7. Save and Print Results
os.makedirs('icphm_2026/datasets', exist_ok=True)
df.to_csv('icphm_2026/datasets/ip_profile_clustered_original.csv', index=False)
original_centroids.to_csv('icphm_2026/datasets/centroid_investigation.csv', index=False)

print("--- Cluster Membership Counts ---")
print(df['cluster_id'].value_counts().sort_index())

print("\n--- Cluster Centroids with Representative IP ---")
print(original_centroids)

print(f"\nSaved investigation summary to 'icphm_2026/datasets/centroid_investigation.csv'")