import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

df = pd.read_csv('datasets/filtered_data.csv')

linkage_array = linkage(df.values, method='average', metric='euclidean')

# saves the cluser labels for k=5 and k=3
for k in [3, 5]:
    labels = fcluster(linkage_array, t=k, criterion='maxclust')
    df[f'cluster_k{k}'] = labels
    print(f'Cluster counts for k={k}:\n{df[f"cluster_k{k}"].value_counts()}\n')

df.to_csv('datasets/clustered_data.csv', index=False)

# 3D scatter plot for k=3
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['total_sessions'],
    df['username_entropy'],
    df['command_entropy'],
    c=df['cluster_k3'],
    cmap=cm.get_cmap('viridis', 3),
    s=50,
    alpha=0.8
)
ax.set_title('3D Scatter Plot of Clusters (k=3)')
ax.set_xlabel('Total Sessions')
ax.set_ylabel('Username Entropy')
ax.set_zlabel('Command Entropy')
cbar = fig.colorbar(scatter, ax=ax, ticks=[1, 2, 3])
cbar.set_label('Cluster Label')
plt.tight_layout()
plt.show()

# 3D scatter plot for k=5
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['total_sessions'],
    df['username_entropy'],
    df['command_entropy'],
    c=df['cluster_k5'],
    cmap=cm.get_cmap('viridis', 5),
    s=50,
    alpha=0.8       
)
ax.set_title('3D Scatter Plot of Clusters (k=5)')
ax.set_xlabel('Total Sessions')
ax.set_ylabel('Username Entropy')
ax.set_zlabel('Command Entropy')
cbar = fig.colorbar(scatter, ax=ax, ticks=[1, 2, 3, 4, 5])
cbar.set_label('Cluster Label')
plt.tight_layout()
plt.show()


# plot figures on axis total session count, username entropy, password entropy with command entropy as color bar

# 3D scatter plot with command entropy as color bar
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['total_sessions'],
    df['username_entropy'],
    df['password_entropy'],
    c=df['command_entropy'],
    cmap='viridis',
    s=50,
    alpha=0.8
)
ax.set_title('3D Scatter Plot with Command Entropy as Color Bar')
ax.set_xlabel('Total Sessions')
ax.set_ylabel('Username Entropy')
ax.set_zlabel('Password Entropy')
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Command Entropy')
plt.tight_layout()
plt.show()

# 3d plotting useranme entropy, password entropy, command entropy with cluster as color bar and total session count as size
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['username_entropy'],
    df['password_entropy'],
    df['command_entropy'],
    c=df['cluster_k5'],
    s=df['total_sessions'] * 2,  # scale size for better visibility
    cmap=cm.get_cmap('viridis', 5),
    alpha=0.6
)
ax.set_title('3D Scatter Plot of Entropies with Cluster Labels and Session Size')
ax.set_xlabel('Username Entropy')
ax.set_ylabel('Password Entropy')
ax.set_zlabel('Command Entropy')
cbar = fig.colorbar(scatter, ax=ax, ticks=[1, 2, 3, 4, 5])
cbar.set_label('Cluster Label')
plt.tight_layout()
plt.show()  

# 3d plotting username entropy, password entropy, command entropy with cluster as color bar and total session count as size
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['username_entropy'],
    df['password_entropy'],
    df['command_entropy'],
    c=df['cluster_k3'],
    s=df['total_sessions'] * 2,  # scale size for better visibility
    cmap=cm.get_cmap('viridis', 3),
    alpha=0.6
)
ax.set_title('3D Scatter Plot of Entropies with Cluster Labels and Session Size')
ax.set_xlabel('Username Entropy')
ax.set_ylabel('Password Entropy')
ax.set_zlabel('Command Entropy')
cbar = fig.colorbar(scatter, ax=ax, ticks=[1, 2, 3])
cbar.set_label('Cluster Label')
plt.tight_layout()
plt.show()  

# 3d plot with username entropy, password entropy, command entropy with total session count as color bar and cluster as size for k=5
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['username_entropy'],
    df['password_entropy'],
    df['command_entropy'],
    c=df['total_sessions'],
    s=df['cluster_k5'] * 20,  # scale size for better visibility
    cmap='viridis',
    alpha=0.6
)
ax.set_title('3D Scatter Plot of Entropies with Session Count and Cluster Size (k=5)')
ax.set_xlabel('Username Entropy')
ax.set_ylabel('Password Entropy')
ax.set_zlabel('Command Entropy')
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Total Sessions')
plt.tight_layout()
plt.show()  

# 3d plot with username entropy, password entropy, command entropy with total session count as color bar and cluster as size for k=3
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['username_entropy'],
    df['password_entropy'],
    df['command_entropy'],
    c=df['total_sessions'],
    s=df['cluster_k3'] * 20,  # scale size for better visibility
    cmap='viridis',
    alpha=0.6
)
ax.set_title('3D Scatter Plot of Entropies with Session Count and Cluster Size (k=3)')
ax.set_xlabel('Username Entropy')
ax.set_ylabel('Password Entropy')
ax.set_zlabel('Command Entropy')
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Total Sessions')
plt.tight_layout()
plt.show()   


# 3d with username entropy, password entropy, command entropy with total session count as color bar and cluster as size for k=3 with size map 

# assign big differences per cluster
size_map = {1: 60, 2: 200, 3: 600}  # you can tweak these
sizes = df['cluster_k3'].map(size_map)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['username_entropy'],
    df['password_entropy'],
    df['command_entropy'],
    c=df['total_sessions'],
    s=sizes,          # use the mapped sizes
    cmap='viridis',
    alpha=0.7,
    edgecolor='k'
)
ax.set_title('3D Scatter: Entropies (color=Sessions, size=Cluster k=3)')
ax.set_xlabel('Username Entropy')
ax.set_ylabel('Password Entropy')
ax.set_zlabel('Command Entropy')

cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Total Sessions')

plt.tight_layout()
plt.show()

# 3d with username entropy, password entropy, command entropy with total session count as color bar and cluster as size for k=5 with size map
# assign big differences per cluster
size_map = {1: 40, 2: 100, 3: 200, 4: 400, 5: 600}  # you can tweak these
sizes = df['cluster_k5'].map(size_map)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['username_entropy'],
    df['password_entropy'],
    df['command_entropy'],
    c=df['total_sessions'],
    s=sizes,          # use the mapped sizes
    cmap='viridis',
    alpha=0.7,
    edgecolor='k'
)
ax.set_title('3D Scatter: Entropies (color=Sessions, size=Cluster k=5)')
ax.set_xlabel('Username Entropy')
ax.set_ylabel('Password Entropy')
ax.set_zlabel('Command Entropy')    
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Total Sessions')
plt.tight_layout()
plt.show()

