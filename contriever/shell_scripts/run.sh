#!/bin/bash

# SLURM directives
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out


echo "--------------------------------"
echo "Set up environment"
echo "--------------------------------"

# Set up directories and environment variables
export TRANSFORMERS_CACHE=/home/huan/scratch/Model_Training/contriever_framework/.cache/huggingface
export HF_HOME=/home/huan/scratch/Model_Training/contriever_framework/.cache/huggingface
export SAVING_DIRECTORY=/home/huan/scratch/Model_Training/contriever_framework

#Load environmental modules
module load python/3.10
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5  cuda/12.2
module load arrow/15.0.1
module load faiss/1.7.4

# Set up the Python virtual environment
virtualenv --no-download $SAVING_DIRECTORY/venv
source $SAVING_DIRECTORY/venv/bin/activate

# Install required Python packages
pip install datasets --no-index
pip install transformers --no-index
pip install torch --no-index
pip install beir --no-index
pip install pandas --no-index
pip install numpy --no-index

echo "--------------------------------"
echo "Environment setup complete."
echo "--------------------------------"

# Data preparation
echo "--------------------------------"
echo "Preparing data..."
echo "--------------------------------"
python $SAVING_DIRECTORY/python_scripts/main.py
echo "--------------------------------"
echo "Data preparation complete."
echo "--------------------------------"

# Document embedding
echo "--------------------------------"
echo "Embedding documents..."
echo "--------------------------------"
chmod u+x $SAVING_DIRECTORY/shell_scripts/embed_documents.sh
$SAVING_DIRECTORY/shell_scripts/embed_documents.sh
echo "--------------------------------"
echo "Document embedding complete."
echo "--------------------------------"

#Retrieval process
echo "--------------------------------"
echo "Starting retrieval..."
echo "--------------------------------"
chmod u+x $SAVING_DIRECTORY/shell_scripts/contriever_retrieval.sh
$SAVING_DIRECTORY/shell_scripts/contriever_retrieval.sh
echo "--------------------------------"
echo "Retrieval complete."
echo "--------------------------------"

echo "All processes completed successfully."

