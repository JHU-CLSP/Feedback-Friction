# RIGID THINKING: LLMs Struggle to Fully Incorporate External Feedback

A research framework for evaluating how large language models incorporate different styles of feedback across multiple reasoning domains.

## Overview

This project implements a unified framework to test large language models' (LLMs) ability to use different types of feedback:

- **Binary feedback**: Simple correct/incorrect signals
- **Self-generated feedback**: Model-generated reflective feedback  
- **Strong-model feedback**: External model-generated feedback

The framework runs multiple iterations of generation and refinement across various datasets (MMLU, MMLU-Pro, GPQA, MATH-500, AIME 2024, PopQA, TriviaQA, arithmetic, and hexadecimal multiplication) to measure iterative self-improvement capabilities.

## Installation

### Prerequisites
- Python 3.9+
- vLLM 0.8.3+ (for model serving)
- OpenAI API key (optional, for strong-model feedback)

### Setup
```bash
git clone https://github.com/JHU-CLSP/Feedback-Friction.git
cd Feedback-Friction
pip install vllm==0.8.3
pip install -r requirements.txt
```

### Environment Configuration
Set your OpenAI API key if using strong-model feedback:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

All experiments are driven by `openai_async_process.py`. The basic command structure is:

```bash
python openai_async_process.py \
    --dataset DATASET \
    --agent_model MODEL_NAME \
    --base_url BASE_URL \
    --ports PORT_LIST \
    --write_file OUTPUT_FILE \
    --iterations NUM_ITERATIONS \
    [FEEDBACK_OPTIONS]
```

### Example Commands

**Basic usage (binary feedback only):**
```bash
python openai_async_process.py \
    --dataset gpqa \
    --agent_model meta-llama/Llama-3.3-70B-Instruct \
    --base_url http://c007 \
    --ports 1233 \
    --write_file gpqa_log.jsonl \
    --iterations 10
```

**Self-generated feedback:**
```bash
python openai_async_process.py \
    --dataset gpqa \
    --agent_model meta-llama/Llama-3.3-70B-Instruct \
    --base_url http://c007 \
    --ports 1233 \
    --write_file gpqa_log.jsonl \
    --iterations 10 \
    --use_feedback
```

**Process-level feedback:**
```bash
python openai_async_process.py \
    --dataset gpqa \
    --agent_model meta-llama/Llama-3.3-70B-Instruct \
    --base_url http://c007 \
    --ports 1233 \
    --write_file gpqa_log.jsonl \
    --iterations 10 \
    --use_feedback \
    --use_process_feedback
```

**Strong-model feedback (requires OpenAI API key):**
```bash
python openai_async_process.py \
    --dataset gpqa \
    --agent_model meta-llama/Llama-3.3-70B-Instruct \
    --base_url http://c007 \
    --ports 1233 \
    --write_file gpqa_log.jsonl \
    --iterations 10 \
    --use_feedback \
    --use_process_feedback \
    --use_openai
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset` | str | `"math"` | Dataset to evaluate (see supported datasets below) |
| `--agent_model` | str | `"meta-llama/Meta-Llama-3-8B-Instruct"` | Model name for vLLM server |
| `--write_file` | str | `"output_arc.jsonl"` | Output file path |
| `--base_url` | str | `"http://c004"` | vLLM server base URL |
| `--ports` | str | `"1233_1234_1235_1236"` | Underscore-separated server ports |
| `--temperature` | float | `0.0` | Sampling temperature |
| `--iterations` | int | `10` | Number of feedback iterations |
| `--proportion` | float | `1.0` | Fraction of dataset to use (0-1) |
| `--use_feedback` | flag | `False` | Enable self-generated feedback |
| `--use_process_feedback` | flag | `False` | Enable process-level feedback |
| `--use_openai` | flag | `False` | Use OpenAI for feedback generation |
| `--shuffle` | flag | `False` | Shuffle MCQ answer choices between iterations |
| `--binary_hint` | flag | `False` | Provide hints about previous incorrect choices |
| `--in_temp` | flag | `False` | Increase temperature each iteration |
| `--best_of_n` | flag | `False` | Enable best-of-n sampling per round |
| `--logprobs` | int | `None` | Number of log probabilities to return |

## Supported Datasets

- **MMLU**: Massive Multitask Language Understanding
- **MMLU-Pro**: Enhanced version of MMLU  
- **GPQA**: Graduate-level Google-Proof Q&A
- **MATH**: MATH-500 mathematical reasoning
- **AIME 2024**: American Invitational Mathematics Examination
- **TriviaQA**: Trivia question answering
- **PopQA**: Popular question answering
- **Custom Simple**: 5-digit decimal multiplication
- **Hex**: 5-digit hexadecimal multiplication

**Deprecated**: GSM8K, GSM8K-Symbolic (no longer supported)

## Feedback Modes

### 1. Binary Feedback (Default)
Provides only correct/incorrect signals after each attempt.

### 2. Self-Generated Feedback (`--use_feedback`)
The model generates its own reflective feedback about errors.

### 3. Process-Level Feedback (`--use_feedback --use_process_feedback`)
Includes detailed reasoning process in feedback generation.

### 4. Strong-Model Feedback (`--use_feedback --use_process_feedback --use_openai`)
Uses OpenAI's models to generate high-quality feedback.

## Output Format

Results are saved as JSONL files with the following fields:

- **question**: Complete interaction history with original question
- **normalized_answer**: Ground truth answer
- **normalized_prediction**: Extracted model prediction  
- **full_response**: Raw model response for current iteration
- **feedback**: Generated feedback (if feedback is enabled)
- **response_probs**: Average log probability per token
- **is_correct**: Whether current iteration is correct
- **iteration**: Current iteration number (starting from 0)

## File Structure

- **`openai_async_process.py`**: Main experiment runner
- **`utils.py`**: Core utilities and dataset handling
- **`manual_hints_5d.py`**: Arithmetic problem hints and feedback
- **`error_analysis.py`**: Feedback following analysis (requires OpenAI API)
- **`check_dis_new.py`**: Output distribution analysis
- **`start_multiple_server_*.sh`**: vLLM server startup scripts

## Citation

Please check out our paper for more implementation details and design choices.