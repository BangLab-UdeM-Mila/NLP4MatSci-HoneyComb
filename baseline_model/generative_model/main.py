import argparse
import os
import sys
sys.path.append(os.environ["SAVING_DIRECTORY"])

from utils.data_preparation import load_dataset, prepare_data
from models import gpt_model, llama_model


MODEL_CHOICES = ['gpt-3.5', 'gpt-4', 'llama2-7b', 'llama3-8b', 'honeybee-7b']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

def main(args):
    assert args.model in MODEL_CHOICES, "illegal model choice, please pick from gpt-3.5, gpt-4, llama2-7b, llama3-8b, honeybee-7b"

    dataset_path = os.path.join(os.environ['SAVING_DIRECTORY'], f"dataset/{args.dataset}/{args.dataset}.parquet" if args.dataset == "sciq" else f"dataset/{args.dataset}/mascqa-eval-with_answer.json")
    dataset = load_dataset(dataset_path, args.dataset)
    data = prepare_data(dataset, args.dataset, limit=args.limit)
    excel_path = f"./{args.model}_{args.dataset}_results.xlsx"
    
    if args.model.startswith("gpt"):
        gpt_model.gpt(OPENAI_API_KEY, args.model, data, excel_path, args.dataset)
    else:
        llama_model.llama(args.model, data, excel_path, args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES, help="Choose the model to use")
    parser.add_argument("--dataset", required=True, choices=['sciq', 'mascqa'], help="Choose the dataset to use")
    parser.add_argument("--limit", type=int, help="Limit the number of test cases")
    args = parser.parse_args()
    main(args)
