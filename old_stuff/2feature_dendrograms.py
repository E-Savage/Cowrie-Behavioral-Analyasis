import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


# load in full unfiltered entropy dataset
df = pd.read_csv('datasets/filtered_data.csv')

# fix dataframe
cols = ['total_sessions', 'username_entropy', 'password_entropy', 'command_entropy']
for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')   

# create dendrograms for different pair feauture sets
feature_sets = [
    ('total_sessions', 'username_entropy'),
    ('total_sessions', 'password_entropy'),
    ('total_sessions', 'command_entropy'),
    ('username_entropy', 'password_entropy'),
    ('username_entropy', 'command_entropy'),
    ('password_entropy', 'command_entropy'),
]

output_dir = 'dendrograms/pairwise'

for feature_set in feature_sets:
    feature1, feature2 = feature_set
    data = df[[feature1, feature2]].dropna().values

    # perform hierarchical clustering
    Z = linkage(data, method='ward')

    # create dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(
        Z, 
        labels=[f"{row[0]:.0f}, {row[1]:.0f}" for row in data], 
        leaf_rotation=90,
        leaf_font_size=8,
        p=50,
        truncate_mode='lastp'
    )
    plt.title(f'Dendrogram for {feature1} and {feature2}')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')

    # save dendrogram
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dendrogram_{feature1}_{feature2}.png', dpi=300)
    plt.close()