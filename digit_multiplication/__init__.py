"""Digit multiplication module for 5-digit and 6-digit numbers in both decimal and hexadecimal."""

from .decimal import provide_multiplication_hints_5d_6d, extract_numbers_and_process_5d_6d
from .hexadecimal import multiply_hex_step_by_step, generate_hex_questions, process_jsonl

__all__ = [
    'provide_multiplication_hints_5d_6d',
    'extract_numbers_and_process_5d_6d', 
    'multiply_hex_step_by_step',
    'generate_hex_questions',
    'process_jsonl'
]