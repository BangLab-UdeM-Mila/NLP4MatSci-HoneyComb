import os

def create_prompt_for_sciq(question, distractors, correct_answer):
    choice_str = "\n    ".join([f"A: {distractors[0]}", f"B: {distractors[1]}", f"C: {distractors[2]}", f"D: {correct_answer}"])
    return f"""
    This is a multiple choice question.
    **Question**: {question}
    **Choices**:
    {choice_str}

    **Please answer with only the letter of the correct answer.**
    """

def extract_baseline_answer_for_sciq(response_content):
    content = response_content.strip()
    return "CORRECT" if content.find("D") != -1 else "INCORRECT"

def create_prompt_for_mascqa(question, qtype):
    MULTIPLE_CHOICE_QUESTION_TAG = ["MCQS", "MATCH", "MCQS-NUM"]
    if qtype in MULTIPLE_CHOICE_QUESTION_TAG:
        question_type = "multiple choice question"
        action = "**Please answer with only the letter of the correct answer.**"
    else:
        question_type = "numerical computation question"
        action = "**Please answer with only the numeric value result.**"
    
    return f"""
    This is {question_type}.
    **Question**: {question}

    {action}
    """

def extract_baseline_answer_for_mascqa(response_content):
    return response_content.strip()

def create_prompt_for_glass(question):
    question_type = "yes or no question"
    action = "**Please answer with only yes or no.**"
    
    return f"""
    This is {question_type}.
    **Question**: Does the given paragraph pertain to glass science?
    **Paragraph**: {question}

    {action}
    """

def extract_baseline_answer_for_glass(response_content):
    return response_content.strip()

def create_prompt_for_sofc_sent(question):
    question_type = "a yes or no question"
    action = "**Please answer with only yes or no.**"
    
    return f"""
    This is {question_type}.
    **Question**: Does the given sentence describe relevant experimental facts?
    **Sentence**: {question}

    {action}
    """