import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

# 1. Load the original IP profile dataset
file_path = 'datasets/ip_entropies.csv'
df = pd.read_csv(file_path)

# 2. Data Cleaning & Pre-processing
# Force numeric types to ensure total_sessions isn't dropped or treated as a string
if 'total_sessions' in df.columns:
    df['total_sessions'] = pd.to_numeric(df['total_sessions'], errors='coerce')

# Identify numerical features and fill NaNs with 0
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(0)
original_numeric_data = df[numeric_cols]

# 3. Create Normalized DataFrame for Clustering
# We normalize to ensure sessions and entropy contribute equally to the distance math
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(original_numeric_data)

# 4. Perform K-Means (k=5)
# Using a fixed random_state for reproducibility in your research
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(normalized_values)
df['cluster_id'] = cluster_labels

# 5. FIND THE CLOSEST ACTUAL PROFILE TO EACH CENTROID
# Instead of averages, we find the real IP record that sits at the center
representative_profiles = []
# Identify the IP column automatically
ip_col = 'ip' if 'ip' in df.columns else df.select_dtypes(include=['object']).columns[0]

for i in range(5):
    # Get indices of all IPs in this specific cluster
    indices = np.where(cluster_labels == i)[0]
    
    # Get the normalized coordinates and the mathematical centroid
    cluster_points = normalized_values[indices]
    centroid = kmeans.cluster_centers_[i].reshape(1, -1)
    
    # Calculate Euclidean distance from every point in cluster to its centroid
    distances = cdist(cluster_points, centroid, 'euclidean')
    
    # Identify the index of the real record with the minimum distance
    closest_point_idx = indices[np.argmin(distances)]
    
    # Extract that real-world row (maintains whole numbers and original values)
    rep_row = df.iloc[closest_point_idx].copy()
    representative_profiles.append(rep_row)

# 6. Convert results to a DataFrame
rep_df = pd.DataFrame(representative_profiles)

# 7. Save and Print Results
os.makedirs('icphm_2026/datasets', exist_ok=True)
df.to_csv('icphm_2026/datasets/ip_profile_clustered_original.csv', index=False)
rep_df.to_csv('icphm_2026/datasets/representative_profiles.csv', index=False)

print("--- Cluster Membership Counts ---")
print(df['cluster_id'].value_counts().sort_index())

print("\n--- Archetypal Profiles (Actual Records Closest to Centroids) ---")
# This prints real whole-number session counts and actual IP addresses
print(rep_df[[ip_col, 'cluster_id'] + list(numeric_cols)].to_string(index=False))

print(f"\nSaved analysis to 'icphm_2026/datasets/representative_profiles.csv'")