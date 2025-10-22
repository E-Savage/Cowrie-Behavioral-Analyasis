import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Load data ---
df = pd.read_csv('datasets/clustered_data.csv')

# --- Select features (do NOT include labels here) ---
feat_cols = ['total_sessions', 'username_entropy', 'password_entropy', 'command_entropy']
X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
X = StandardScaler().fit_transform(X)

# --- UMAP 2D ---
umap2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
emb2d = umap2d.fit_transform(X)

# --- Plot 2D colored by k=5 cluster labels ---
labels_k5 = pd.to_numeric(df['cluster_k5'], errors='coerce').fillna(-1).astype(int)
unique_k5 = np.sort(labels_k5.unique())

plt.figure(figsize=(10, 7))
# Discrete colormap with enough distinct colors
cmap5 = ListedColormap(plt.cm.tab10.colors[:max(5, len(unique_k5))])
sc = plt.scatter(emb2d[:, 0], emb2d[:, 1], c=labels_k5, cmap=cmap5, s=60, alpha=0.9, edgecolors='k')
plt.title('UMAP (2D) • Color = Cluster k=5', fontsize=15)
plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
cbar = plt.colorbar(sc)
cbar.set_label('Cluster Label (k=5)')
plt.tight_layout()
# plt.savefig('plots/october_17_2025/umap_2d_k5.png', dpi=200)
plt.show()

# --- UMAP 3D ---
umap3d = umap.UMAP(n_components=3, n_neighbors=20, min_dist=0.05, metric='euclidean', random_state=42)
emb3d = umap3d.fit_transform(X)

# --- Plot 3D colored by k=3 cluster labels ---
labels_k3 = pd.to_numeric(df['cluster_k3'], errors='coerce').fillna(-1).astype(int)
unique_k3 = np.sort(labels_k3.unique())
cmap3 = ListedColormap(plt.cm.tab10.colors[:max(3, len(unique_k3))])

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(emb3d[:, 0], emb3d[:, 1], emb3d[:, 2],
               c=labels_k3, cmap=cmap3, s=60, alpha=0.9, edgecolor='k')
ax.set_title('UMAP (3D) • Color = Cluster k=3', fontsize=14)
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2'); ax.set_zlabel('UMAP 3')
fig.colorbar(p, ax=ax, shrink=0.6, pad=0.1).set_label('Cluster Label (k=3)')
ax.view_init(elev=25, azim=35)
plt.tight_layout()
# plt.savefig('plots/october_17_2025/umap_3d_k3.png', dpi=200)
plt.show()