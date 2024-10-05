import os
import re


HF_HOME = os.environ["HF_HOME"]


def get_model_config(model_name):
    if model_name == "honeybee-7b":
        return f"{HF_HOME}/yahma/llama-7b-hf"
    elif model_name == "llama2-7b":
        return f"{HF_HOME}/meta-llama/Llama-2-7b-chat-hf"
    elif model_name == "llama3-8b":
        return f"{HF_HOME}/meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == "gpt-3.5":
        return "gpt-3.5-turbo-0125"
    elif model_name == "gpt-4":
        return "gpt-4o"
    return None


def get_model_nickname(model_id):
    if model_id == f"{HF_HOME}/yahma/llama-7b-hf":
        return "honeybee"
    elif model_id == f"{HF_HOME}/meta-llama/Llama-2-7b-chat-hf":
        return "llama2"
    elif model_id == f"{HF_HOME}/meta-llama/Meta-Llama-3-8B-Instruct":
        return "llama3"
    else:
        if re.search(r"3.5", model_id):
            return "gpt3.5"
        else:
            return "gpt4"