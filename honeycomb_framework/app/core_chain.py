import logging
import os
from typing import List, Generator, Any, Optional

import pandas as pd
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from utils.save_and_read_excel import save_to_excel, read_recent_recalled_or_retrieved_documents
from utils.get_model_config import get_model_nickname
from utils.langchain_utils import create_langchain_documents
from utils.model_utils import llama_prompt_modifier

from matrag_framework.retriever_layer import retriever
from matrag_framework.recall_layer.recall import RecallLayer
from matrag_framework.recall_layer.knowledge_base_db import KnowledgeBase


question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text verbatim.
{context}
Question: {question}
Relevant text, if any:"""
QUESTION_PROMPT = PromptTemplate(template=question_prompt_template, input_variables=["context", "question"])

combine_prompt_template = """
Instruction:
Given the following extracted parts of a long document and a question, create a final answer.
!!!Make sure you follow the following rules!!!

    1.  Chain of Thought: When summarizing, perform a step-by-step problem-solving approach to ensure clarity and logical flow.
    2.  Answer the question with your own knowledge if the answer cannot be determined based on the provided information.
    3.  Special Note for Computational Questions:
    •   If any document states, “this is the answer to the problem using predefined tools” with a numerical answer, use that answer unless it is unreasonable.
    •   If the stated answer is unreasonable, use your own knowledge to perform the computation again and answer the computational question. You must provide a numerical answer in all cases.
    4.  !!Answer Format!!:
    •   For multiple-choice questions, end with "The answer is" followed by the letter of the correct choice (A, B, C, D, or E) without any symbols or periods.
    •   For computational questions, end with "The answer is" followed the numerical answer without any units symbols or periods.

QUESTION: {question}
=========
{summaries}
=========
If the information above do not conclusively answer the question:
    • For computational questions, perform a new computation *step by step* based on the general knowledge about the topic.
    • For multiple-choice questions, deduce the most likely answer based on the general knowledge about the topic.
    • Make sure you follow the !!Answer Format!! when answering questions
!!!If an answer cannot be determine, you must provide your best educated guess - either a choice of letter for multiple-choice question or a numeric value for a computation question. Other answers are prohibited.!!!
FINAL ANSWER:"""
COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["summaries", "question"])


def map_step(documents: List[str], question: str, qa_pipeline) -> List[str]:
    # Iterate over documents
    results = []
    for doc in documents:
        prompt = question_prompt_template.format(context=doc, question=question)
        result = llama_prompt_modifier(prompt, qa_pipeline)
        results.append(result)
    return results


def reduce_step(answers: List[str], question: str, summary_pipeline) -> str:
    # Gather all answers
    gathered_answers = "\n".join(answers)
    final_prompt = combine_prompt_template.format(summaries=gathered_answers, question=question)
    result = llama_prompt_modifier(final_prompt, summary_pipeline)
    return result


def map_reduce(documents: List[str], question: str, qa_pipeline, summary_pipeline) -> str:
    # Perform the map step
    answers = map_step(documents, question, qa_pipeline)
    
    # Perform the reduce step
    final_answer = reduce_step(answers, question, summary_pipeline)
    
    return final_answer


def custom_load_qa_chain(
        llm: BaseLanguageModel = None,
        verbose: Optional[bool] = None,
        callback_manager: Optional[BaseCallbackManager] = None,
        langchain: bool = True,
        documents=None,
        question=None,
        qa_pipeline=None,
        summary_pipeline=None,
        **kwargs: Any,
) -> BaseCombineDocumentsChain:
    if langchain:
        return load_qa_chain(
            llm,
            chain_type="map_reduce",
            verbose=verbose,
            question_prompt=QUESTION_PROMPT,
            combine_prompt=COMBINE_PROMPT,
            callback_manager=callback_manager, 
            **kwargs
        )
    else:
        return map_reduce(documents, question, qa_pipeline, summary_pipeline)
