import json
import re
import random


def add_hex(a: str, b: str) -> str:
    """
    Add two hexadecimal numbers (given as strings) digit by digit in base 16.
    """
    if len(b) > len(a):
        a, b = b, a
    a_rev, b_rev = a[::-1], b[::-1]
    carry = 0
    result_digits = []
    for i in range(len(a_rev)):
        x = int(a_rev[i], 16)
        y = int(b_rev[i], 16) if i < len(b_rev) else 0
        s = x + y + carry
        carry = s // 16
        result_digits.append(hex(s % 16)[2:].upper())
    if carry:
        result_digits.append(hex(carry)[2:].upper())
    return ''.join(result_digits[::-1])


def multiply_hex_digit(hex_num: str, d: str) -> str:
    """
    Multiply a hexadecimal number (hex_num) by a single hex digit d.
    """
    rev = hex_num[::-1]
    carry = 0
    result_digits = []
    d_val = int(d, 16)
    for char in rev:
        p = int(char, 16) * d_val + carry
        carry = p // 16
        result_digits.append(hex(p % 16)[2:].upper())
    if carry:
        result_digits.append(hex(carry)[2:].upper())
    return ''.join(result_digits[::-1])


def multiply_hex_step_by_step(a: str, b: str):
    """
    Multiply two hexadecimal numbers using only base 16 operations.
    Returns (steps, explanation) with normalized intermediate sums.
    """
    b_rev = b[::-1]
    steps = {"partial_products": [], "intermediate_sums": [], "final_result": ""}
    current_sum = "0"
    explanation = f"Multiplying {a} by {b} in base 16:\n"
    for i, digit in enumerate(b_rev):
        partial = multiply_hex_digit(a, digit)
        shifted = partial + "0" * i
        steps["partial_products"].append({
            "digit_of_b": digit,
            "partial_product": partial,
            "shifted_partial_product": shifted,
            "shift_amount": i
        })
        current_sum = add_hex(current_sum, shifted)
        normalized = current_sum.lstrip('0') or '0'
        steps["intermediate_sums"].append(normalized)
        explanation += (
            f"Step {i+1}: Multiply {a} by '{digit}' → Partial product: {partial}, "
            f"Shifted: {shifted}, Intermediate Sum: {normalized}\n"
        )
    final_norm = current_sum.lstrip('0') or '0'
    steps["final_result"] = final_norm
    explanation += f"Final Product (hex) = {final_norm}"
    return steps, explanation


def generate_hex_questions(output_file: str, num_questions: int = 1000, digit_length: int = 5):
    """
    Generate a JSONL file of multiplication questions with random "decimal-looking" hex numbers.
    Each number uses only digits 0–9 but is interpreted in base 16.
    """
    with open(output_file, "w") as outfile:
        for _ in range(num_questions):
            num1 = ''.join(random.choice('0123456789') for _ in range(digit_length))
            num2 = ''.join(random.choice('0123456789') for _ in range(digit_length))
            question = f"Calculate the following question: {num1} * {num2}."
            record = {"question": question, "range": f"[{num1}, {num2}]"}
            outfile.write(json.dumps(record) + "\n")


def process_jsonl(input_file: str, output_file: str):
    """
    Read questions from a JSONL, compute hex products, and write detailed explanations.
    """
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            question = data.get("question", "")
            nums = re.findall(r"\b[0-9A-Fa-f]+\b", question)
            if len(nums) < 2:
                print(f"Skipping invalid question: {question}")
                continue
            n1, n2 = nums[0].upper(), nums[1].upper()
            steps, explanation = multiply_hex_step_by_step(n1, n2)
            computed = steps["final_result"]
            expected = hex(int(n1, 16) * int(n2, 16))[2:].upper()
            verification = (computed == expected)
            output_record = {
                "question": (
                    f"Calculate the following question, where each number is represented in base 16: "
                    f"{n1} * {n2}."
                ),
                "answer": expected,
                "range": data.get("range"),
                "Explanation": explanation,
                "Verification": verification
            }
            outfile.write(json.dumps(output_record) + "\n")


if __name__ == "__main__":
    # Generate and process 5-digit hex questions
    input_file = "hex_questions_5d.jsonl"
    generate_hex_questions(input_file, num_questions=1000, digit_length=5)
    output_file = "hex5d.jsonl"
    process_jsonl(input_file, output_file)
    print(f"Generated and processed 5-digit hex questions → {output_file}")