import os
import sys
sys.path.append(os.environ["SAVING_DIRECTORY"])
from tqdm import tqdm

from utils.model_utils import create_llama_pipeline, llama_prompt_modifier
from utils.get_model_config import get_model_config
from utils.data_utils import *
from utils.save_and_read_excel import *


HF_HOME = os.environ["HF_HOME"]


def llama(model_name, data, excel_path, dataset_type):

    model_id = get_model_config(model_name)

    pipeline = create_llama_pipeline(model_id)
    predictions = []
    
    for item in tqdm(data, desc="running generation"):
        if dataset_type == "sciq":
            question, distractors, correct_answer = item
            prompt = create_prompt_for_sciq(question, distractors, correct_answer)
            extract_answer_fn = extract_baseline_answer_for_sciq
            save_to_excel = save_to_excel_sciq_baseline
            max_new_tokens = 3
        elif dataset_type == "glass":
            index, question, correct_answer = item
            prompt = create_prompt_for_glass(question)
            extract_answer_fn = extract_baseline_answer_for_glass
            save_to_excel = save_to_excel_glass_baseline
            max_new_tokens = 32
        elif dataset_type == "sofc_sent":
            index, question, correct_answer = item
            prompt = create_prompt_for_sofc_sent(question)
            extract_answer_fn = extract_baseline_answer_for_glass
            save_to_excel = save_to_excel_glass_baseline 
            max_new_tokens = 512
        else:
            qid, question, qtype, correct_answer, topic = item
            prompt = create_prompt_for_mascqa(question, qtype)
            extract_answer_fn = extract_baseline_answer_for_mascqa
            save_to_excel = save_to_excel_mascqa_baseline
            max_new_tokens = 512
        
        outputs = llama_prompt_modifier(prompt, pipeline, max_new_tokens)

        predicted_answer = extract_answer_fn(outputs)
        predictions.append(predicted_answer)
        save_to_excel(excel_path, item, predicted_answer, outputs)
    
    return predictions
