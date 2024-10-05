def extract_baseline_answer_for_sciq(response):
    content = response.strip()
    return "CORRECT" if content.find("D") != -1 else "INCORRECT"