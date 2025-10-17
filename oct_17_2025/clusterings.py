import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import sys

df = pd.read_csv('datasets/ip_entropies.csv')

# fix dataframe 
cols = ['total_sessions', 'username_entropy', 'password_entropy', 'command_entropy']
for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# correlation matrix
corr_matrix = df[['total_sessions', 'username_entropy', 'password_entropy', 'command_entropy']].corr()
print("Corr Matrix:")
print(corr_matrix)

# plot correlation matrix 
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45 )
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.title('Corralation Matrix')
plt.tight_layout()
plt.show()

# correlation matrix of total_sessions, username_entropy, password_entropy
corr_matrix_subset = df[['total_sessions', 'username_entropy', 'password_entropy']].corr()
print("Subset Corr Matrix:")
print(corr_matrix_subset)
plt.imshow(corr_matrix_subset, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(corr_matrix_subset)), corr_matrix_subset.columns, rotation=45 )
plt.yticks(range(len(corr_matrix_subset)), corr_matrix_subset.columns)
plt.title('Subset Corralation Matrix')
plt.tight_layout()
plt.show()

# correlation of username_entropy, password_entropy, command_entropy
corr_matrix_entropy = df[['username_entropy', 'password_entropy', 'command_entropy']].corr()
print("Entropy Corr Matrix:")
print(corr_matrix_entropy)
plt.imshow(corr_matrix_entropy, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(corr_matrix_entropy)), corr_matrix_entropy.columns, rotation=45 )
plt.yticks(range(len(corr_matrix_entropy)), corr_matrix_entropy.columns)
plt.title('Entropy Corralation Matrix')
plt.tight_layout()
plt.show()  

# scatter plots
# total_sessions vs username_entropy
plt.scatter(df['total_sessions'], df['username_entropy'], alpha=0.5)
plt.title('Total Sessions vs Username Entropy')
plt.xlabel('Total Sessions')
plt.ylabel('Username Entropy')
plt.tight_layout()
plt.show()

# total_sessions vs password_entropy
plt.scatter(df['total_sessions'], df['password_entropy'], alpha=0.5)
plt.title('Total Sessions vs Password Entropy')
plt.xlabel('Total Sessions')
plt.ylabel('Password Entropy')
plt.tight_layout()
plt.show()

# total_sessions vs command_entropy
plt.scatter(df['total_sessions'], df['command_entropy'], alpha=0.5)
plt.title('Total Sessions vs Command Entropy')
plt.xlabel('Total Sessions')
plt.ylabel('Command Entropy')
plt.tight_layout()
plt.show()


#-----------------------------------------
# hierarchical clustering
sys.setrecursionlimit(10000)

mask = df[cols].notna().all(axis=1)

clustering_data = df.loc[mask, cols].values
linkage_array = linkage(clustering_data, method='average', metric='euclidean')
dendrogram(
    linkage_array,
    truncate_mode='lastp',   # or 'level'
    p=30,                    # show only last 30 merged clusters
    show_leaf_counts=True,
    leaf_rotation=90,
)
plt.title('Dendrogram of IP Profiles')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# cut outlier out of the clusters
distances = linkage_array[:, 2]
sorted_distances = np.sort(distances)
gaps = np.diff(sorted_distances)
k = np.argmax(gaps)
threshold = (sorted_distances[k] + sorted_distances[k + 1]) / 2
print(f"Chosen threshold for clustering: {threshold}")

labels = fcluster(linkage_array, t=threshold, criterion='distance')

# create the column and fill only clustered rows
df['cluster'] = pd.NA
df.loc[mask, 'cluster'] = labels
df['cluster'] = df['cluster'].astype('Int64') # keeps NA for unclustered rows

cluster_counts = df.loc[mask, 'cluster'].value_counts()
print("Cluster counts:")
print(cluster_counts)

min_size = 5
small_clusters = cluster_counts[cluster_counts < min_size].index
df_filtered = df[~df['cluster'].isin(small_clusters)]
print(f"Filtered out clusters smaller than {min_size}. New dataset size: {len(df_filtered)}")

# make the new dendrogram with extreme outliers removed
cols = ['total_sessions', 'username_entropy', 'password_entropy', 'command_entropy']

# removing any NAN just for safety 
filtered_data = df_filtered[cols].dropna().values

# recompute linkage for the new dendrogram
linkage_array_filtered = linkage(filtered_data, method='average', metric='euclidean')
plt.figure(figsize=(10, 7))
dendrogram(
    linkage_array_filtered,
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

# produce new scatter plots with filtered data 

# sessions vs username_entropy
plt.scatter(df_filtered['total_sessions'], df_filtered['username_entropy'], alpha=0.5)
plt.title('Filtered: Total Sessions vs Username Entropy')
plt.xlabel('Total Sessions')
plt.ylabel('Username Entropy')
plt.tight_layout()
plt.show() 

# sessions vs password_entropy
plt.scatter(df_filtered['total_sessions'], df_filtered['password_entropy'], alpha=0.5)
plt.title('Filtered: Total Sessions vs Password Entropy')
plt.xlabel('Total Sessions')
plt.ylabel('Password Entropy')
plt.tight_layout()
plt.show()

# sessions vs command_entropy
plt.scatter(df_filtered['total_sessions'], df_filtered['command_entropy'], alpha=0.5)
plt.title('Filtered: Total Sessions vs Command Entropy')
plt.xlabel('Total Sessions')
plt.ylabel('Command Entropy')
plt.tight_layout()
plt.show() 


# command entropy vs username_entropy
plt.scatter(df_filtered['command_entropy'], df_filtered['username_entropy'], alpha=0.5)
plt.title('Filtered: Command Entropy vs Username Entropy')
plt.xlabel('Command Entropy')
plt.ylabel('Username Entropy')
plt.tight_layout()
plt.show()

# command entropy vs password_entropy
plt.scatter(df_filtered['command_entropy'], df_filtered['password_entropy'], alpha=0.5)
plt.title('Filtered: Command Entropy vs Password Entropy')
plt.xlabel('Command Entropy')
plt.ylabel('Password Entropy')
plt.tight_layout()
plt.show()

# username_entropy vs password_entropy
plt.scatter(df_filtered['username_entropy'], df_filtered['password_entropy'], alpha=0.5)
plt.title('Filtered: Username Entropy vs Password Entropy')
plt.xlabel('Username Entropy')
plt.ylabel('Password Entropy')
plt.tight_layout()
plt.show()



