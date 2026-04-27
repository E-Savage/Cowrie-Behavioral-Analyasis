import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import sys
import time

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
INPUT_CSV = "./datasets/ip_entropies.csv"  
OUTPUT_DIR = "robust_k_comparison_results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

script_start = time.time()

try:
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} records from {INPUT_CSV}")
except FileNotFoundError:
    print(f"Error: {INPUT_CSV} not found.")
    sys.exit(1)

# ==========================================
# 2. PREPROCESSING
# ==========================================
IP_COL       = 'src_ip'
SESSIONS_COL = 'total_sessions' 
USER_ENTROPY = 'username_entropy'
PASS_ENTROPY = 'password_entropy'
CMD_ENTROPY  = 'command_entropy'

# Set IP as index
df = df.set_index(IP_COL)

# Force numeric to handle potential "object" errors
df[SESSIONS_COL] = pd.to_numeric(df[SESSIONS_COL], errors='coerce')

# Log-transform for that 1 massive exponential outlier
df['sessions_log'] = np.log1p(df[SESSIONS_COL])

# Select features and drop NaNs
features_list = ['sessions_log', USER_ENTROPY, PASS_ENTROPY, CMD_ENTROPY]
X = df[features_list].dropna()

# Execution settings
IS_TEST = len(sys.argv) > 1 and sys.argv[1] == 'test'
SEEDS_PER_K = 5 if IS_TEST else 100 
K_RANGE = range(2, 11)

# ==========================================
# 3. PHASE 1: COMPREHENSIVE GRID SEARCH
# ==========================================
print(f"Phase 1: Testing k=2-10 for both Scaling Methods ({SEEDS_PER_K} seeds each)...")
comparison_data = []

for k in K_RANGE:
    for seed in range(SEEDS_PER_K):
        # --- Test Standardization ---
        X_std = StandardScaler().fit_transform(X)
        km_std = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(X_std)
        score_std = silhouette_score(X_std, km_std.labels_)
        
        # --- Test Normalization ---
        X_norm = MinMaxScaler().fit_transform(X)
        km_norm = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(X_norm)
        score_norm = silhouette_score(X_norm, km_norm.labels_)
        
        comparison_data.append({
            'k': k,
            'seed': seed,
            'Standardization': score_std,
            'Normalization': score_norm
        })

# Save raw iteration data
raw_results_df = pd.DataFrame(comparison_data)
raw_results_df.to_csv(f"{OUTPUT_DIR}/raw_k_comparison_data.csv", index=False)

# ==========================================
# 4. FINDING THE WINNING PAIR (FIXED)
# ==========================================
# Calculate means and std devs across seeds
summary = raw_results_df.groupby('k').agg(['mean', 'std']).reset_index()

# Fix: Mapping the 7 resulting columns (k + 2 per feature)
# Columns are: k, seed_mean, seed_std, std_mean, std_dev, norm_mean, norm_dev
summary.columns = ['k', 's_m', 's_s', 'std_mean', 'std_dev', 'norm_mean', 'norm_dev']

# Find absolute best (Method, K)
if summary['std_mean'].max() >= summary['norm_mean'].max():
    best_method = "Standardization"
    best_k = int(summary.loc[summary['std_mean'].idxmax(), 'k'])
    best_score = summary['std_mean'].max()
else:
    best_method = "Normalization"
    best_k = int(summary.loc[summary['norm_mean'].idxmax(), 'k'])
    best_score = summary['norm_mean'].max()

print(f"Optimal Configuration: {best_method} with k={best_k} (Mean Silhouette: {best_score:.4f})")

# ==========================================
# 5. FINAL CLUSTERING & EXPORT
# ==========================================
final_scaler = StandardScaler() if best_method == "Standardization" else MinMaxScaler()
X_final = final_scaler.fit_transform(X)
final_km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X_final)

ip_mapping = pd.DataFrame({
    'src_ip': X.index,
    'cluster': final_km.labels_,
    'method_used': best_method,
    'optimal_k': best_k,
    'raw_sessions': df.loc[X.index, SESSIONS_COL]
})
ip_mapping.to_csv(f"{OUTPUT_DIR}/final_assignments_k{best_k}_{best_method}.csv", index=False)

# ==========================================
# 6. VISUALIZATION & REPORTING
# ==========================================
plt.figure(figsize=(10, 6))
plt.errorbar(summary['k'], summary['std_mean'], yerr=summary['std_dev'], 
             label='Standardization', marker='o', capsize=5, color='blue', linewidth=2)
plt.errorbar(summary['k'], summary['norm_mean'], yerr=summary['norm_dev'], 
             label='Normalization', marker='s', capsize=5, color='green', linewidth=2)

plt.title(f'Scaling Robustness across K-Values (Winner: k={best_k})', fontsize=12)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Mean Silhouette Score (+/- Std Dev)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/k_comparison_scaling_plot.png")

script_end = time.time()
with open(f"{OUTPUT_DIR}/phd_validation_report.txt", "w") as f:
    f.write(f"Ph.D. Comprehensive Clustering Validation\n{'='*45}\n")
    f.write(f"Winning Method: {best_method}\n")
    f.write(f"Optimal K:      {best_k}\n")
    f.write(f"Mean Score:     {best_score:.4f}\n")
    f.write(f"Total Time:     {script_end - script_start:.2f}s\n")

print(f"Workflow complete. Results saved in '{OUTPUT_DIR}'.")