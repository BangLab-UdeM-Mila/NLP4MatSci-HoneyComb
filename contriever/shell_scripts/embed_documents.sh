#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out

python $SAVING_DIRECTORY/contriever/generate_passage_embeddings.py \
--model_name_or_path facebook/contriever \
--output_dir $SAVING_DIRECTORY/output/contriever_embeddings \
--passages $SAVING_DIRECTORY/output/support_text.jsonl \
--shard_id 0 \
--num_shards 1 \