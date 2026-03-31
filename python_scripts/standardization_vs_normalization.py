import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os, sys
import time

# setup output directory
INPUT_CSV = "./datasets/ip_entropies.csv"
OUTPUT_DIR = "./python_scripts/clustering_results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

script_timer = time.time()

try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: {INPUT_CSV} not found. Ensure the file is in the script directory.")
    sys.exit(1)

# column mapping
IP_COL = "src_ip" 
SESSIONS_COL = "total_sessions"
USER_ENTROPY  = "username_entropy" 
PASS_ENTROPY = "password_entropy"
CMD_ENTROPY = "command_entropy"

# just gives an index
df = df.set_index(IP_COL)

# ensures that everything is numeric 
df[SESSIONS_COL] = pd.to_numeric(df[SESSIONS_COL], errors='coerce')

# perform log transform on possible super high session counts 
df['sessions_log'] = np.log1p(df[SESSIONS_COL])

# feature selection
features_list = ['sessions_log', USER_ENTROPY, PASS_ENTROPY, CMD_ENTROPY]
X = df[features_list].dropna()

# identify the single massive outlier for your records
top_ip = X['sessions_log'].idxmax()
top_val = df.loc[top_ip, SESSIONS_COL]
print(f"Top Outlier Detected: IP {top_ip} with {top_val} sessions.")

# run with robustness in mind
if len(sys.argv) > 1 and sys.argv[1] == 'test':
    n_iterations = 3
else:
    n_iterations = 100

seeds = np.arange(n_iterations)
results_log = []

for seed in seeds:
    # standardization
    X_std = StandardScaler().fit_transform(X)
    km_std = KMeans(n_clusters=4, n_init=1, random_state=seed).fit_predict(X_std)
    score_std = silhouette_score(X_std, km_std)

    # normalization
    X_norm = MinMaxScaler().fit_transform(X)
    km_norm = KMeans(n_clusters=4, n_init=1, random_state=seed).fit_predict(X_norm)
    score_norm = silhouette_score(X_norm, km_norm)

    results_log.append({
        'seed': seed,
        'standardization_silhouette': score_std,
        'normalization_silhouette': score_norm
    })


# the results

results_df = pd.DataFrame(results_log)
results_df.to_csv(f"{OUTPUT_DIR}/robustness_comparison_results.csv", index=False)

# calculate means to determine the better method statistically 
mean_std = results_df['standardization_silhouette'].mean()
mean_norm = results_df['normalization_silhouette'].mean()

if mean_std >= mean_norm:
    winner_name = "Standardization"
    final_scaler = StandardScaler()
    winner_score = mean_std
else:
    winner_name = "Normalization"
    final_scaler = MinMaxScaler()
    winner_score = mean_norm

print(f"Selection complete. {winner_name} seems to be the most robust method")

# run final clusering using the winnining method for assignments 
X_final_scaled = final_scaler.fit_transform(X)
final_km = KMeans(n_clusters=4, n_init=10, random_state=42)
final_labels = final_km.fit_predict(X_final_scaled)

# save the values
ip_mapping = pd.DataFrame({
    'src_ip': X.index,
    'cluster': final_labels,
    'raw_sessions': df.loc[X.index, SESSIONS_COL]
})

ip_mapping.to_csv(f"{OUTPUT_DIR}/final_ip_cluster_assignments.csv", index=False)

# save summary stats
stats_text = (
    f"Robustness Summary (k=4)\n"
    f"{'='*40}\n"
    f"Standardization Mean: {mean_std:.4f} (Std: {results_df['standardization_silhouette'].std():.4f})\n"
    f"Normalization Mean: {mean_norm:.4f} (Std: {results_df['normalization_silhouette'].std():.4f})\n"
    f"WINNER: {winner_name}"
)
with open(f"{OUTPUT_DIR}/summary_stats.txt", "w") as f:
    f.write(stats_text)

script_end = time.time()
total_duration = script_end - script_timer

# save the distribution plot
plt.figure(figsize=(10, 6))
plt.hist(results_df["standardization_silhouette"], bins=15, alpha=0.5, label='Standardization', color='blue')
plt.hist(results_df["normalization_silhouette"], bins=15, alpha=0.5, label='Normalization', color='green')
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/robustness_distribution.png")

print(f"All files saved to {OUTPUT_DIR}. CSV with 100 iterations is ready as well")
print(f"Done. Total time: {total_duration:.2f}s")
