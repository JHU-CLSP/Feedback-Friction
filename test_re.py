import re

def extract_answer_fixed(dataset_name, prediction):
    if dataset_name == "mmlu":  # extract the formatted answer
        regex1 = re.compile(r"answer is \(?\(([A-D])\)?\)")
        if (match := regex1.search(prediction)):
            return match.group(1).upper()
        regex2 = re.compile(r"\.*\[aA\]nswer:\s*\(([A-D])\)")
        if (match := regex2.search(prediction)):
            return match.group(1).upper()
        regex3 = re.compile(r"answer is \(?([A-D])\)?", re.IGNORECASE)
        if (match := regex3.search(prediction)):
            return match.group(1).upper()
        return "X"  # represent not found
    
    elif dataset_name == "mmlu_pro":  # extract the formatted answer
        regex1 = re.compile(r"answer is \(?\(([A-J])\)?\)")
        if (match := regex1.search(prediction)):
            return match.group(1).upper()
        regex2 = re.compile(r"\.*\[aA\]nswer:\s*\(([A-J])\)")
        if (match := regex2.search(prediction)):
            return match.group(1).upper()
        regex3 = re.compile(r"answer is \(?([A-J])\)?", re.IGNORECASE)
        if (match := regex3.search(prediction)):
            return match.group(1).upper()
        return "X"  # represent not found
    
    elif dataset_name == "gpqa":
        regex1 = re.compile(r"answer is \(?\(([A-D])\)?\)")
        if (match := regex1.search(prediction)):
            return match.group(1).upper()
        regex2 = re.compile(r"\.*\[aA\]nswer:\s*\(([A-D])\)")
        if (match := regex2.search(prediction)):
            return match.group(1).upper()
        regex3 = re.compile(r"answer is \(?([A-D])\)?", re.IGNORECASE)
        if (match := regex3.search(prediction)):
            return match.group(1).upper()
        return "X"  # represent not found

# Test cases
test_cases = [
    ("mmlu", "The answer is (A)."),  # should return 'A'
    ("mmlu", "answer is (B)"),  # should return 'B'
    ("mmlu", "[aA]nswer: (C)"),  # should return 'C'
    ("mmlu", "Answer is (d)"),  # should return 'D' (case insensitive check)
    ("mmlu", "No answer given."),  # should return 'X'

    ("mmlu_pro", "The answer is (F)."),  # should return 'F'
    ("mmlu_pro", "answer is (J)"),  # should return 'J'
    ("mmlu_pro", "[aA]nswer: (H)"),  # should return 'H'
    ("mmlu_pro", "Answer is (i)"),  # should return 'I' (case insensitive check)
    ("mmlu_pro", "No valid answer given."),  # should return 'X'

    ("gpqa", "The answer is (C)."),  # should return 'C'
    ("gpqa", "answer is (D)"),  # should return 'D'
    ("gpqa", "[aA]nswer: (A)"),  # should return 'A'
    ("gpqa", "Answer is (b)"),  # should return 'B' (case insensitive check)
    ("gpqa", "No valid answer available."),  # should return 'X'
]
# Rerun tests
test_results_fixed = [(dataset, prediction, extract_answer_fixed(dataset, prediction)) for dataset, prediction in test_cases]

print(test_results_fixed)
