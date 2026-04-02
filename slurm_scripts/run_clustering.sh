#!/bin/bash

#SBATCH --job-name=honeypot_clustering    # Job name
#SBATCH --output=logs/job_%j.out         # Standard output log (%j = JobID)
#SBATCH --error=logs/job_%j.err          # Standard error log
#SBATCH --partition=cpu                  # Use the CPU partition
#SBATCH --nodes=1                        # Run on a single node
#SBATCH --ntasks=1                       # Run a single task
#SBATCH --cpus-per-task=4                # Request 4 CPU cores (good for parallel KMeans)
#SBATCH --mem=16G                        # Request 16GB of RAM
#SBATCH --time=04:00:00                  # Max walltime (4 hours - plenty for 100 seeds)
#SBATCH --mail-type=END,FAIL             # Email notification when job ends or fails
#SBATCH --mail-user=esavage1@umassd.edu # Replace with your UMass email

# 1. Create a logs directory if it doesn't exist
mkdir -p logs

# 2. Load necessary modules (Unity usually requires loading Python)
module load python/3.11.7  # Use the version matching your zhoulab venv

# 3. Activate your virtual environment
# Replace this path with the actual path to your venv
source /home/esavage1_umassd_edu/Cowrie-Behavioral-Analyasis/venv/bin/activate

# 4. Run the script
# We pass the path to the script. 
# Make sure your CSV path inside the Python script is correct!
python3 ./python_scripts/standardization_vs_normalization.py

echo "Job completed successfully at $(date)"