"""Decimal multiplication hints for 5-digit and 6-digit numbers."""

import re

def provide_multiplication_hints_5d_6d(num1, num2, out=None):
    """Provide step-by-step multiplication hints for 5-digit and 6-digit numbers using distributive property."""
    # Break numbers into significant parts (split at thousands)
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
    output_lines.append("Step2: Then we apply the distributive property:")
    output_lines.append(f"{num1} x {num2} = ({num1_high} + {num1_low}) x ({num2_high} + {num2_low})")
    output_lines.append("Compute partial products:")
    output_lines.append(f"Step3: {num1_high} x {num2_high} = {partial_1}")
    output_lines.append(f"Step4: {num1_high} x {num2_low} = {partial_2}")
    output_lines.append(f"Step5: {num1_low} x {num2_high} = {partial_3}")
    output_lines.append(f"Step6: {num1_low} x {num2_low} = {partial_4}")
    output_lines.append(f"Step7: Sum all partial products we have the answer: {total}")
    output_lines.append(f"Total: {partial_1} + {partial_2} + {partial_3} + {partial_4} = {total}\n")
    
    # Write to file if specified
    if out is not None:
        with open(out, "a") as f:
            f.write("\n".join(output_lines))
            f.write("\n\n")
    
    return ("\n".join(output_lines), [str(partial_1), str(partial_2), str(partial_3), str(partial_4)])

def extract_numbers_and_process_5d_6d(question):
    """Extract numbers from multiplication question and process them."""
    # Pattern for 5-6 digit numbers
    pattern = r'(\d{5,6})\s*\*\s*(\d{5,6})'
    match = re.search(pattern, question)
    
    if match:
        num1, num2 = int(match.group(1)), int(match.group(2))
        hint, partial_products = provide_multiplication_hints_5d_6d(num1, num2)
        return hint
    else:
        return "Error: Could not extract valid 5-6 digit numbers from the question."