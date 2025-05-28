import random
import re
import json

def provide_multiplication_hints(num1, num2, out=None): # same for 5 and 6 digits
    # Break numbers into significant parts
    num1_high = (num1 // 1000) * 1000
    num1_low = num1 % 1000
    
    num2_high = (num2 // 1000) * 1000
    num2_low = num2 % 1000
    
    # Compute partial products
    partial_1 = num1_high * num2_high
    partial_2 = num1_high * num2_low
    partial_3 = num1_low * num2_high
    partial_4 = num1_low * num2_low
    
    # Sum up results
    total = partial_1 + partial_2 + partial_3 + partial_4
    
    # Generate output lines
    output_lines = []
    output_lines.append(f"The original question is:\nCalculate the following question: {num1} * {num2}.")
    output_lines.append(f"Step1: After breaking down the numbers: {num1} = {num1_high} + {num1_low}, {num2} = {num2_high} + {num2_low}")
    output_lines.append("Stwp2: Then we apply the distributive property:")
    output_lines.append(f"{num1} x {num2} = ({num1_high} + {num1_low}) x ({num2_high} + {num2_low})")
    output_lines.append("Compute partial products:")
    output_lines.append(f"Step3: {num1_high} x {num2_high} = {partial_1}")
    output_lines.append(f"Step4: {num1_high} x {num2_low} = {partial_2}")
    output_lines.append(f"Step5: {num1_low} x {num2_high} = {partial_3}")
    output_lines.append(f"Step6: {num1_low} x {num2_low} = {partial_4}")
    output_lines.append(f"Step7: Sum all partial products we have the answer: {total}")
    output_lines.append(f"Total: {partial_1} + {partial_2} + {partial_3} + {partial_4} = {total}\n")
    # output_lines.append(f"Correctness: {total == num1 * num2}")
    
    # Write to file if specified
    if out is not None:
        with open(out, "a") as f:
            f.write("\n".join(output_lines))
            f.write("\n\n")
    
    return ("\n".join(output_lines), [str(partial_1), str(partial_2), str(partial_3), str(partial_4)])

def provide_multiplication_hints_4d(num1, num2, out=None): # revise
    # Break numbers into significant parts
    num1_high = (num1 // 100) * 100
    num1_low = num1 % 100
    
    num2_high = (num2 // 100) * 100
    num2_low = num2 % 100
    
    # Compute partial products
    partial_1 = num1_high * num2_high
    partial_2 = num1_high * num2_low
    partial_3 = num1_low * num2_high
    partial_4 = num1_low * num2_low
    
    # Sum up results
    total = partial_1 + partial_2 + partial_3 + partial_4
    
    # Generate output lines
    output_lines = []
    output_lines.append(f"Calculate the following question: {num1} * {num2}.")
    output_lines.append(f"Break down the numbers: {num1} = {num1_high} + {num1_low}, {num2} = {num2_high} + {num2_low}")
    output_lines.append("Apply distributive property:")
    output_lines.append(f"{num1} x {num2} = ({num1_high} + {num1_low}) x ({num2_high} + {num2_low})")
    output_lines.append("Compute partial products:")
    output_lines.append(f"{num1_high} x {num2_high} = {partial_1}")
    output_lines.append(f"{num1_high} x {num2_low} = {partial_2}")
    output_lines.append(f"{num1_low} x {num2_high} = {partial_3}")
    output_lines.append(f"{num1_low} x {num2_low} = {partial_4}")
    output_lines.append("Sum up the results:")
    output_lines.append(f"Total: {partial_1} + {partial_2} + {partial_3} + {partial_4} = {total}")
    # output_lines.append(f"Correctness: {total == num1 * num2}")
    
    # Write to file if specified
    if out is not None:
        with open(out, "a") as f:
            f.write("\n".join(output_lines))
            f.write("\n\n")
    
    return "\n".join(output_lines)

def provide_multiplication_hints_78d(num1, num2, out=None): # revise
    # Break numbers into significant parts
    num1_high = (num1 // 10000) * 10000
    num1_low = num1 % 10000
    
    num2_high = (num2 // 10000) * 10000
    num2_low = num2 % 10000
    
    # Compute partial products
    partial_1 = num1_high * num2_high
    partial_2 = num1_high * num2_low
    partial_3 = num1_low * num2_high
    partial_4 = num1_low * num2_low
    
    # Sum up results
    total = partial_1 + partial_2 + partial_3 + partial_4
    
    # Generate output lines
    output_lines = []
    output_lines.append(f"Calculate the following question: {num1} * {num2}.")
    output_lines.append(f"Break down the numbers: {num1} = {num1_high} + {num1_low}, {num2} = {num2_high} + {num2_low}")
    output_lines.append("Apply distributive property:")
    output_lines.append(f"{num1} x {num2} = ({num1_high} + {num1_low}) x ({num2_high} + {num2_low})")
    output_lines.append("Compute partial products:")
    output_lines.append(f"{num1_high} x {num2_high} = {partial_1}")
    output_lines.append(f"{num1_high} x {num2_low} = {partial_2}")
    output_lines.append(f"{num1_low} x {num2_high} = {partial_3}")
    output_lines.append(f"{num1_low} x {num2_low} = {partial_4}")
    output_lines.append("Sum up the results:")
    output_lines.append(f"Total: {partial_1} + {partial_2} + {partial_3} + {partial_4} = {total}")
    # output_lines.append(f"Correctness: {total == num1 * num2}")
    
    # Write to file if specified
    if out is not None:
        with open(out, "a") as f:
            f.write("\n".join(output_lines))
            f.write("\n\n")
    
    return "\n".join(output_lines)

def provide_multiplication_hints_9d(num1, num2, out=None): # revise
    # Break numbers into significant parts
    num1_high = (num1 // 100000) * 100000
    num1_low = num1 % 100000
    
    num2_high = (num2 // 100000) * 100000
    num2_low = num2 % 100000
    
    # Compute partial products
    partial_1 = num1_high * num2_high
    partial_2 = num1_high * num2_low
    partial_3 = num1_low * num2_high
    partial_4 = num1_low * num2_low
    
    # Sum up results
    total = partial_1 + partial_2 + partial_3 + partial_4
    
    # Generate output lines
    output_lines = []
    output_lines.append(f"Calculate the following question: {num1} * {num2}.")
    output_lines.append(f"Break down the numbers: {num1} = {num1_high} + {num1_low}, {num2} = {num2_high} + {num2_low}")
    output_lines.append("Apply distributive property:")
    output_lines.append(f"{num1} x {num2} = ({num1_high} + {num1_low}) x ({num2_high} + {num2_low})")
    output_lines.append("Compute partial products:")
    output_lines.append(f"{num1_high} x {num2_high} = {partial_1}")
    output_lines.append(f"{num1_high} x {num2_low} = {partial_2}")
    output_lines.append(f"{num1_low} x {num2_high} = {partial_3}")
    output_lines.append(f"{num1_low} x {num2_low} = {partial_4}")
    output_lines.append("Sum up the results:")
    output_lines.append(f"Total: {partial_1} + {partial_2} + {partial_3} + {partial_4} = {total}")
    # output_lines.append(f"Correctness: {total == num1 * num2}")
    
    # Write to file if specified
    if out is not None:
        with open(out, "a") as f:
            f.write("\n".join(output_lines))
            f.write("\n\n")
    
    return "\n".join(output_lines)

def extract_initial_question(raw_text):
    """
    Given a raw question string, this function:
    1. Strips off any text after "\n\nPrevious Answer:".
    2. Strips off any text after " Choices: " if present.
    3. Replaces any remaining newlines with spaces.
    4. Returns the cleaned question on one line.
    """
    # truncate at "\n\nPrevious Answer:" if that marker exists.
    marker_prev_answer = "\n\nPrevious Answer:"
    if marker_prev_answer in raw_text:
        question_part = raw_text.split(marker_prev_answer)[0]
    else:
        question_part = raw_text

    return question_part

def extract_numbers_and_process_5d(question):
    question = extract_initial_question(question)
    match = re.findall(r"\d{5}", question)
    if len(match) >= 2:
        num1, num2 = int(match[0]), int(match[1])
        # print(f"Processing: {num1} x {num2}")
        return provide_multiplication_hints(num1, num2) # return a tuple of (feedback, list of partial ground truths)
    else:
        return "Error: Could not extract two five-digit numbers from the question."
    
def extract_numbers_and_process_6d(question):
    question = extract_initial_question(question)
    match = re.findall(r"\d{6}", question)
    if len(match) >= 2:
        num1, num2 = int(match[0]), int(match[1])
        # print(f"Processing: {num1} x {num2}")
        return provide_multiplication_hints(num1, num2)
    else:
        return "Error: Could not extract two six-digit numbers from the question."
    
def extract_numbers_and_process_4d(question):
    question = extract_initial_question(question)
    match = re.findall(r"\d{4}", question)
    if len(match) >= 2:
        num1, num2 = int(match[0]), int(match[1])
        # print(f"Processing: {num1} x {num2}")
        return provide_multiplication_hints_4d(num1, num2)
    else:
        return "Error: Could not extract two four-digit numbers from the question."

def extract_numbers_and_process_7d(question):
    question = extract_initial_question(question)
    match = re.findall(r"\d{7}", question)
    if len(match) >= 2:
        num1, num2 = int(match[0]), int(match[1])
        # print(f"Processing: {num1} x {num2}")
        return provide_multiplication_hints_78d(num1, num2)
    else:
        return "Error: Could not extract two seven-digit numbers from the question."

def extract_numbers_and_process_8d(question):
    question = extract_initial_question(question)
    match = re.findall(r"\d{8}", question)
    if len(match) >= 2:
        num1, num2 = int(match[0]), int(match[1])
        # print(f"Processing: {num1} x {num2}")
        return provide_multiplication_hints_78d(num1, num2)
    else:
        return "Error: Could not extract two seven-digit numbers from the question."
    
def extract_numbers_and_process_9d(question):
    question = extract_initial_question(question)
    match = re.findall(r"\d{9}", question)
    if len(match) >= 2:
        num1, num2 = int(match[0]), int(match[1])
        # print(f"Processing: {num1} x {num2}")
        return provide_multiplication_hints_9d(num1, num2)
    else:
        return "Error: Could not extract two seven-digit numbers from the question."

'''
file_path = "/scratch/dkhasha1/bzhang90/Self-InPerfect/digits_buckets/multiplication_questions_9d.jsonl"
with open(file_path, 'r') as file:
    questions = [json.loads(line) for line in file]
    for ques in questions:
        extract_numbers_and_process_9d(ques["question"])
'''