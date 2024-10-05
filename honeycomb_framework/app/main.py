import argparse
import os
import sys
sys.path.append(os.environ["SAVING_DIRECTORY"])
sys.path.append(os.environ["SAVING_DIRECTORY"] + "/contriever_framework/.cache")
import logging

from utils.save_and_read_excel import save_to_excel
from utils.get_model_config import get_model_nickname
from utils.get_model_config import get_model_config
from utils.model_utils import create_llama_pipeline

from matrag_framework.app.qa_chain import DocChatter
from matrag_framework.recall_layer.knowledge_base_db import KnowledgeBase


recall_logger = logging.getLogger('RecallLayer')
recall_logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


COMPONENT_TESTING_CHOICES = ["recall_only", "retrieve_only", "llm_generation_only"]
MODEL_CHOICES = ['gpt-3.5', 'gpt-4', 'llama2-7b', 'llama3-8b', 'honeybee-7b']
SAVING_DIRECTORY = os.environ["SAVING_DIRECTORY"]
HF_HOME = os.environ["HF_HOME"]
TOOL_SAVE_PATH = os.environ['SAVING_DIRECTORY'] + f"/matrag_framework/output/tools_output/tools_output.xlsx"
KB_PATH = SAVING_DIRECTORY + "/matrag_framework/recall_layer/databases/knowledge_base.json"
NUM_RECALLED = 10
NUM_RETRIEVED = 3


def honeycomb(query, model_id, pipeline=None, component_testing=None):

    PREDICTION_SAVING_PATH = f"{SAVING_DIRECTORY}/matrag_framework/output/prediction_output/user_test/{get_model_nickname(model_id)}.xlsx"
    
    if not query:
        raise ValueError("No query argument provided, query argument must be provided for one_time_generation task")
    
    knowledge_base = KnowledgeBase(KB_PATH, recall_logger)
    
    result = DocChatter.map_reduce_chat(
        n_recall=NUM_RECALLED,
        n_retrieved=NUM_RETRIEVED,
        knowledge_base=knowledge_base,
        query=query,
        agent_model=model_id,
        pipeline=pipeline,
        id=None,
        qtype=None,
        component_testing=component_testing
    )

    if not component_testing or component_testing == "llm_generation_only":
        save_to_excel(PREDICTION_SAVING_PATH, 
                column_values=[query, result],
                column_names=["question", "generated_response"])
    
    print("program finished running >>>>>>>>")
    print(f"results saved to {PREDICTION_SAVING_PATH}")


def main(args):
    assert args.model in MODEL_CHOICES, "illegal model choice, please pick from gpt-3.5, gpt-4, llama2-7b, llama3-8b, honeybee-7b"
    print("program start running >>>>>>>>>")
    model_id = get_model_config(args.model)

    if not args.model.startswith("gpt"):
        pipeline = create_llama_pipeline(model_id)
    else:
        pipeline = None
    
    if args.component_testing:
        honeycomb(query=args.query, model_id=model_id, pipeline=pipeline, component_testing=args.component_testing)
    else:
        honeycomb(query=args.query, model_id=model_id, pipeline=pipeline)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=False, type=str)
    parser.add_argument("--model", required=True, type=str, choices=MODEL_CHOICES, help="pick the large language model used by honeycomb")
    parser.add_argument("--component_testing", 
                        required=False, 
                        type=str, 
                        choices=COMPONENT_TESTING_CHOICES,
                        help="component testing is used for debugging and unit testing purpose")
    args = parser.parse_args()
    main(args)