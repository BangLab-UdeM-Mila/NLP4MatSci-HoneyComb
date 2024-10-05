import logging
import os
from re import S
from typing import List

import pandas as pd
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline

from utils.save_and_read_excel import save_to_excel, read_recent_recalled_or_retrieved_documents
from utils.get_model_config import get_model_nickname
from utils.langchain_utils import create_langchain_documents

from matrag_framework.retriever_layer.retriever import retriever
from matrag_framework.app.core_chain import custom_load_qa_chain
from matrag_framework.recall_layer.recall import RecallLayer
from matrag_framework.recall_layer.knowledge_base_db import KnowledgeBase


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SAVING_DIRECTORY = os.environ["SAVING_DIRECTORY"]


class DocChatter(object):
    enable_debug = True

    @classmethod
    def enable_debug_mode(cls, is_enable: bool) -> None:
        cls.enable_debug = is_enable
        #langchain.debug = is_enable
        logging.info(f"Debug mode set to {'enabled' if is_enable else 'disabled'}.")

    @classmethod
    def map_reduce_chat(cls, 
                        n_recall: int, 
                        n_retrieved: int, 
                        query: str, 
                        agent_model: str, 
                        knowledge_base: KnowledgeBase,
                        dataset: str = "user_test",
                        id: str = None, 
                        qtype: str = None, 
                        correct_answer: str = None,
                        pipeline: str = None,
                        component_testing: str = None):
        
        RECALLED_DOCUMENTS_SAVING_PATH = f"{SAVING_DIRECTORY}/matrag_framework/output/recalled_documents_output/{dataset}/recalled_documents.xlsx"
        RETRIEVED_DOCUMENTS_SAVING_PATH = f"{SAVING_DIRECTORY}/matrag_framework/output/retrieved_documents_output/{dataset}/retrieved_documents.xlsx"
        
        model_nickname = get_model_nickname(agent_model)
        
        # Recall Layer
        if not component_testing or component_testing == "recall_only":

            recaller = RecallLayer(knowledge_base=knowledge_base, 
                                    n_recall=n_recall, 
                                    agent_model=agent_model, 
                                    dataset=dataset, 
                                    correct_answer=correct_answer)

            recalled_documents = recaller.query(query, id=id, qtype=qtype)
            save_to_excel(RECALLED_DOCUMENTS_SAVING_PATH, column_values=[id, query, recalled_documents, correct_answer] ,column_names=["id", "question", "recalled_documents", "correct_answer"])

        if not component_testing or component_testing == "retrieve_only":
            (id, question, recalled_documents, correct_answer) = read_recent_recalled_or_retrieved_documents(RECALLED_DOCUMENTS_SAVING_PATH)
            recalled_documents = eval(recalled_documents)

            # Retriever Layer
            retrieved_documents = retriever(question, recalled_documents=recalled_documents, n_retrieved=n_retrieved, dataset=dataset)

            save_to_excel(RETRIEVED_DOCUMENTS_SAVING_PATH, column_values=[id, question, retrieved_documents, correct_answer], column_names=["id", "question", "retrieved_documents", "correct_answer"])

        if not component_testing or component_testing == "llm_generation_only":

            (id, question, retrieved_documents, correct_answer) = read_recent_recalled_or_retrieved_documents(RETRIEVED_DOCUMENTS_SAVING_PATH)
            retrieved_documents = eval(retrieved_documents)
        
            if model_nickname.startswith("gpt"):
                langchain = True
                docs = create_langchain_documents(retrieved_documents)
            
                if len(docs) == 0:
                    print("No documents found after contriever process finishes")

                logging.debug(f"Retrieved top {n_retrieved} documents for query.")

    
                agent = ChatOpenAI(model=agent_model, temperature=0.0)

                nchain = custom_load_qa_chain(llm=agent, langchain=langchain)

                summarized_response = nchain({"input_documents": docs, "question": query})
        
                return summarized_response["output_text"]
            
            else:
                langchain = False
                result = custom_load_qa_chain(documents=retrieved_documents, question=query, qa_pipeline=pipeline, summary_pipeline=pipeline, langchain=langchain)

                return result