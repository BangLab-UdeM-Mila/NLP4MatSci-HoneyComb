from re import L
import pandas as pd
import json
import random

def load_dataset(dataset_path, dataset_type):
    if dataset_type == "sciq":
        return pd.read_parquet(dataset_path)
    elif dataset_type == "mascqa" or dataset_type == "glass" or dataset_type == "sofc_sent":
        with open(dataset_path, "r") as f:
            data = json.load(f)
        return data
    else:
        raise ValueError("Unknown dataset type")

def prepare_data(dataset, dataset_type, limit=None):
    results = []
    if dataset_type == "sciq":
        for idx in range(min(limit, len(dataset)) if limit else len(dataset)):
            question_data = dataset.iloc[idx]
            question = question_data['question']
            correct_answer = question_data['correct_answer']
            distractors = [question_data['distractor3'], question_data['distractor2'], question_data['distractor1']]
            results.append((question, distractors, correct_answer))
    elif dataset_type == "mascqa":
        count = 0
        for category in dataset.keys():
            qids = dataset[category]['qids']
            qtypes = dataset[category]['qstr']
            questions = dataset[category]['questions']
            correct_answers = dataset[category]['correct_answers']
            for qid, qtype, question, correct_answer in zip(qids, qtypes, questions, correct_answers):
                results.append((qid, question, qtype, correct_answer, category))
                count += 1
                if limit and count >= limit:
                    break
            if limit and count >= limit:
                break
    elif dataset_type == "glass":
        sample_index = random.sample(dataset['index'], 500)
        questions = [x[0] for x in dataset['data']]
        correct_answers = [x[1] for x in dataset['data']]
        for i in sample_index:
            results.append((i, questions[i], correct_answers[i]))
    elif dataset_type == "sofc_sent":
        questions = [x[0] for x in dataset['data']]
        correct_answers = [x[1] for x in dataset['data']]
        for i in range(len(questions)):
            results.append((dataset['index'][i], questions[i], correct_answers[i]))
    return results