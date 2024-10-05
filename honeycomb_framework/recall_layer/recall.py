import logging
import os
from datetime import datetime as dt
import json

import pandas as pd

from utils.save_and_read_excel import save_to_excel
from utils.get_model_config import get_model_nickname
from utils.truncate_to_max_tokens import truncate_to_max_tokens
from matrag_framework.recall_layer.knowledge_base_db import KnowledgeBase
from matrag_framework.recall_layer.tool_agent import ToolExecutor


logger = logging.getLogger('RecallLayer')
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class RecallLayer:

    def enable_debug_mode(self, is_enable: bool) -> None:
        logger.setLevel(logging.DEBUG) if is_enable else None
        logger.info(f"Debug mode set to {'enabled' if is_enable else 'disabled'}")

    def __init__(self, 
                 knowledge_base, 
                 n_recall, 
                 agent_model, 
                 dataset, 
                 correct_answer):
        self.knowledge_base = knowledge_base
        self.tool_agent = ToolExecutor()
        self.n_recall = n_recall
        self.dataset = dataset
        self.model_nickname = get_model_nickname(agent_model)
        self.correct_answer = correct_answer

    def query(self, question, id=None, qtype=None):

        kb_docs = self.knowledge_base.search(question, n_recall=self.n_recall)
        
        logger.debug(f"knowledge_base_db returns a {type(kb_docs)} object")
        logger.debug(f"containing {len(kb_docs)} elements") if isinstance(kb_docs, list) else None

        TOOLS_OUTPUT_SAVE_PATH = os.environ['SAVING_DIRECTORY'] + f"/matrag_framework/output/tools_output/{self.dataset}/{self.model_nickname}_tools_output.xlsx"
       
        try:
            tool_response = self.tool_agent.answer_question(question)

            tool_prompt = f"The answer to the question by running pre-defined tools, is: {tool_response['output']}. The following are the tools used and the responses: {tool_response['intermediate_steps']}"
            
            predicted_reponse = tool_response['output']

            intermediate_step = tool_response['intermediate_steps']
        
        except Exception as e:
            print(f"An error has occurred with tool calling step: {e}")
            predicted_reponse = ""
            tool_prompt = ""
            intermediate_step = ""

        column_values = [id, qtype, question, predicted_reponse, intermediate_step]
        column_names = ["Question Info", "Question Type", "Question", "Tool Response", 'intermediate_function_calls']
        
        if self.dataset == "sciq":
            column_values.append(self.correct_answer)
            column_names.append("Correct Answer")
        
        save_to_excel(TOOLS_OUTPUT_SAVE_PATH, column_values, column_names)

        # Truncate each string in kb_docs to have a maximum 
        # These 2700 and 5000 are used to avoid exceeding a limit of 32,767 in a cell in excel workbook
        # TODO: replace saving to excel with json
        kb_docs = [doc[:2700] for doc in kb_docs]

        # Encode each string to handle special characters
        kb_docs = [json.dumps(doc) for doc in kb_docs]

        combined_docs =  kb_docs + [tool_prompt[:5000]]

        print(tool_prompt)

        return combined_docs