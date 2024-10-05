import json
import logging
import os
import subprocess
from typing import List

from contriever_framework.python_scripts.utils import read_jsonl
from matrag_framework.retriever_layer.data_processor import DataProcessor


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


SAVING_DIRECTORY = os.environ["SAVING_DIRECTORY"]

# Path to the directory containing the passage_retrieval.py script
contriever_scripts_directory = f'{SAVING_DIRECTORY}/contriever_framework/contriever'

# Construct the command
contriever_docs_embed_command = """
python $SAVING_DIRECTORY/contriever_framework/contriever/generate_passage_embeddings.py \
--model_name_or_path $HF_HOME/facebook/contriever \
--output_dir $SAVING_DIRECTORY/matrag_framework/output/contriever_embeddings_{batch_n} \
--passages $SAVING_DIRECTORY/matrag_framework/output/support_text_{batch_n}.jsonl \
--shard_id 0 \
--num_shards 1 
"""


contriever_retrieval_command = """
python $SAVING_DIRECTORY/contriever_framework/contriever/passage_retrieval.py \
 --model_name_or_path $HF_HOME/facebook/contriever \
 --passages $SAVING_DIRECTORY/matrag_framework/output/support_text_{batch_n}.jsonl \
 --passages_embeddings $SAVING_DIRECTORY/matrag_framework/output/contriever_embeddings_{batch_n}/* \
 --data $SAVING_DIRECTORY/matrag_framework/output/data_{batch_n}.jsonl \
 --output_dir $SAVING_DIRECTORY/matrag_framework/output/data_with_supporting_documents \
 --n_docs 100 \
 --top_k_returned_docs {top_k_returned_docs}
"""


def retriever(question: str, 
              recalled_documents: List[str], 
              n_retrieved: int,
              dataset: str,
              batch_n: int = 0):

    config = {
        'SAVING_DIRECTORY': os.environ["SAVING_DIRECTORY"],
        'CACHE_DIR': os.environ['HF_HOME'],
        'OUTPUT_DIR': os.path.join(os.environ["SAVING_DIRECTORY"], "matrag_framework/output")
    }

    if dataset == "sciq":
        config["DATASET_DIRECTORY"] = config['SAVING_DIRECTORY'] +"/dataset/Sciq/sciq.parquet"

    elif dataset == "mascqa":
        config["DATASET_DIRECTORY"] = config['SAVING_DIRECTORY'] +"/dataset/MaScQA/mascqa-eval.json"

    contriever_processor = DataProcessor(config)
    contriever_processor.load_dataset(question=question, recalled_documents=recalled_documents)
    contriever_processor.process_data(batch_n=batch_n)

    # Execute the command
    subprocess.run(contriever_docs_embed_command.format(batch_n=batch_n), shell=True, cwd=contriever_scripts_directory)

    subprocess.run(contriever_retrieval_command.format(batch_n=batch_n, top_k_returned_docs=n_retrieved), shell=True, cwd=contriever_scripts_directory)

    query_and_documents = read_jsonl(config['OUTPUT_DIR']+f"/data_with_supporting_documents/data_{batch_n}_with_top_k_docs.jsonl")
    
    results = [json.dumps(doc_info["text"]) for doc_info in query_and_documents[0]['ctxs']]

    return results

    
     

    

    



