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

def provide_multiplication_hints_7d(num1, num2, out=None): # revise
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

def extract_numbers_and_process_5d(question):
    match = re.findall(r"\d{5}", question)
    if len(match) >= 2:
        num1, num2 = int(match[0]), int(match[1])
        # print(f"Processing: {num1} x {num2}")
        return provide_multiplication_hints(num1, num2)
    else:
        return "Error: Could not extract two five-digit numbers from the question."
    
def extract_numbers_and_process_6d(question):
    match = re.findall(r"\d{6}", question)
    if len(match) >= 2:
        num1, num2 = int(match[0]), int(match[1])
        # print(f"Processing: {num1} x {num2}")
        return provide_multiplication_hints(num1, num2)
    else:
        return "Error: Could not extract two six-digit numbers from the question."
    
def extract_numbers_and_process_4d(question):
    match = re.findall(r"\d{4}", question)
    if len(match) >= 2:
        num1, num2 = int(match[0]), int(match[1])
        # print(f"Processing: {num1} x {num2}")
        return provide_multiplication_hints_4d(num1, num2)
    else:
        return "Error: Could not extract two four-digit numbers from the question."

def extract_numbers_and_process_7d(question):
    match = re.findall(r"\d{7}", question)
    if len(match) >= 2:
        num1, num2 = int(match[0]), int(match[1])
        # print(f"Processing: {num1} x {num2}")
        return provide_multiplication_hints_7d(num1, num2)
    else:
        return "Error: Could not extract two seven-digit numbers from the question."


'''
file_path = "/scratch/dkhasha1/bzhang90/Self-InPerfect/digits_buckets/multiplication_questions_4d.jsonl"
with open(file_path, 'r') as file:
    questions = [json.loads(line) for line in file]
    for ques in questions:
        extract_numbers_and_process_4d(ques["question"])
'''