import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import sys
import os

df = pd.read_csv('datasets/filtered_data.csv')

df_no_sessions = df[['username_entropy', 'password_entropy', 'command_entropy']]
df_no_uname = df[['total_sessions', 'password_entropy', 'command_entropy']]
df_no_pass = df[['total_sessions', 'username_entropy', 'command_entropy']]
df_no_command = df[['total_sessions', 'username_entropy', 'password_entropy']]

linkage_no_sessions = linkage(df_no_sessions.values, method='average', metric='euclidean')
linkage_no_uname = linkage(df_no_uname.values, method='average', metric='euclidean')
linkage_no_pass = linkage(df_no_pass.values, method='average', metric='euclidean')
linkage_no_command = linkage(df_no_command.values, method='average', metric='euclidean')

# make dendrograms for each case

# no total_sessions
plt.figure(figsize=(10, 7))
dendrogram(
    linkage_no_sessions,
    truncate_mode='lastp',
    p=30,
    show_leaf_counts=True,
    leaf_rotation=90,
)
plt.title('Dendrogram without Total Sessions')
plt.xlabel('Filtered Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('dendrograms/dendrogram_no_sessions.png', dpi=300)
plt.close()


# no username_entropy
plt.figure(figsize=(10, 7))
dendrogram(
    linkage_no_uname,
    truncate_mode='lastp',
    p=30,
    show_leaf_counts=True,
    leaf_rotation=90,
)
plt.title('Dendrogram without Username Entropy')
plt.xlabel('Filtered Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('dendrograms/dendrogram_no_username_entropy.png', dpi=300)
plt.close()

# no password_entropy
plt.figure(figsize=(10, 7))
dendrogram(
    linkage_no_pass,
    truncate_mode='lastp',
    p=30,
    show_leaf_counts=True,
    leaf_rotation=90,
)
plt.title('Dendrogram without Password Entropy')
plt.xlabel('Filtered Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('dendrograms/dendrogram_no_password_entropy.png', dpi=300)
plt.close()

# no command_entropy
plt.figure(figsize=(10, 7))
dendrogram(
    linkage_no_command,
    truncate_mode='lastp',
    p=30,
    show_leaf_counts=True,
    leaf_rotation=90,
)
plt.title('Dendrogram without Command Entropy')
plt.xlabel('Filtered Index')
plt.ylabel('Distance')
plt.tight_layout() 
plt.savefig('dendrograms/dendrogram_no_command_entropy.png', dpi=300)
plt.close()


# evaluate clustering metrics for each case
from sklearn.metrics import silhouette_score, davies_bouldin_score



 