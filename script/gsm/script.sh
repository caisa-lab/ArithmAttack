#!/bin/bash
#SBATCH --job-name=gsm
#SBATCH --output=logs/%j_gsm_output.log
#SBATCH --error=logs/%j_gsm_error.log
#SBATCH --partition=sgpu_long
#SBATCH --time=50:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12

# activate the judge environment
source venv/bin/activate
python script/gsm/script_run.py