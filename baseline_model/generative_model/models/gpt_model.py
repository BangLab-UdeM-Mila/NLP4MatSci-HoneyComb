from openai import OpenAI
from tqdm import tqdm

from utils.get_model_config import get_model_config
from utils.data_utils import *
from utils.save_and_read_excel import save_to_excel_sciq_baseline, save_to_excel_mascqa_baseline, save_to_excel_glass_baseline

def get_openai_client(api_key):
    return OpenAI(api_key=api_key)

def gpt(api_key, model_name, data, excel_path, dataset_type):
    model = get_model_config(model_name)
    client = get_openai_client(api_key)
    predictions = []
    
    for item in tqdm(data, desc="running generation"):
        if dataset_type == "sciq":
            question, distractors, correct_answer = item
            prompt = create_prompt_for_sciq(question, distractors, correct_answer)
            extract_answer_fn = extract_baseline_answer_for_sciq
            save_to_excel = save_to_excel_sciq_baseline
        elif dataset_type == "glass":
            index, question, correct_answer = item
            prompt = create_prompt_for_glass(question)
            extract_answer_fn = extract_baseline_answer_for_glass
            save_to_excel = save_to_excel_glass_baseline
        elif dataset_type == "sofc_sent":
            index, question, correct_answer = item
            prompt = create_prompt_for_sofc_sent(question)
            extract_answer_fn = extract_baseline_answer_for_glass
            save_to_excel = save_to_excel_glass_baseline 
        else:
            qid, question, qtype, correct_answer, topic = item
            prompt = create_prompt_for_mascqa(question, qtype)
            extract_answer_fn = extract_baseline_answer_for_mascqa
            save_to_excel = save_to_excel_mascqa_baseline
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        predicted_answer = extract_answer_fn(response.choices[0].message.content)
        predictions.append(predicted_answer)
        save_to_excel(excel_path, item, predicted_answer, response.choices[0].message.content)
    
    return predictions