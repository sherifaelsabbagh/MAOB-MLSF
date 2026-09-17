#!/bin/bash
#SBATCH --job-name=svm_mfplec_tuning
#SBATCH --account=p337
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=svm_mfplec_tuning_%j.out
#SBATCH --error=svm_mfplec_tuning_%j.err

set -eo pipefail
source ~/.bashrc
conda activate protocol-env

python svm_regression_mfplec_tuning.py
