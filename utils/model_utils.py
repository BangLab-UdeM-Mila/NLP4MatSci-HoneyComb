import logging
import os
import re
import string
import sys
sys.path.append(os.environ["SAVING_DIRECTORY"] + "/contriever_framework/.cache")

import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, PreTrainedTokenizerFast

from utils.get_model_config import get_model_nickname
from huggingface.peft.src.peft import PeftModel


HF_HOME = os.environ["HF_HOME"]


def sanitize(input_data):
    if isinstance(input_data, str):
        return ''.join(filter(lambda x: x in string.printable, input_data))
    return input_data


def llama_prompt_modifier(prompt, pipeline, max_new_tokens=512):
    counter = 0 
    while counter <= 5:
        prompt_template = sanitize(prompt)
        messages = [
            {"role": "system", "content": "You are a material scientist who knows all the material science knowledge"},
            {"role": "user", "content": prompt_template},
        ]

        try:
            
            prompt = pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )

            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )
            # Decode the generated tokens
            generated_text = outputs[0]["generated_text"][len(prompt):]

            if len(generated_text) == 0:
                counter += 1
                logging.warning(f"No response generated for question ID {id}, retrying... ({counter}/5)")
                generated_text = ""
            else:
                break
        except Exception as e:
            logging.error(f"Error generating response for question ID {id}: {e}")
            counter += 1
            continue
    return generated_text


def create_llama_pipeline(model_id):
    print(model_id)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_nickname = get_model_nickname(model_id=model_id)
    if model_nickname == "llama3":
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id)

    model = LlamaForCausalLM.from_pretrained(model_id,
                                            load_in_8bit=False,
                                            torch_dtype=torch.float16,
                                            device_map="auto")
    if model_nickname == "honeybee":    
        model = PeftModel.from_pretrained(
            model,
            f"{HF_HOME}/Bang-UdeM-Mila/HoneyBee7b",
            device_map={"": device}
        )

        model.half()

    pipeline = transformers.pipeline("text-generation", 
                                        model=model, 
                                        tokenizer=tokenizer,
                                        model_kwargs={"torch_dtype": torch.bfloat16},
                                        device_map="auto",)
    
    return pipeline