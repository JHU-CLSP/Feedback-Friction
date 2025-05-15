# Self-InPrefect
Repository for our paper RIGID THINKING: LLMs Struggle to Fully Incorporate External Feedback

The main file for our experiments is openai_async_process.py which has the following command:

| Option                   | Type    | Default                                 | Description                                                 |
| ------------------------ | ------- | --------------------------------------- | ----------------------------------------------------------- |
| `--dataset`              | `str`   | `"math"`                                | Which dataset to use (e.g., `math`, `gpqa`, `pop_qa`, etc.) |
| `--agent_model`          | `str`   | `"meta-llama/Meta-Llama-3-8B-Instruct"` | Model name or path for generation (vLLM or HuggingFace)     |
| `--write_file`           | `str`   | `"output_arc.jsonl"`                    | Path to write the per-example JSONL output                  |
| `--base_url`             | `str`   | `"http://c004"`                         | Base URL for your vLLM server                               |
| `--ports`                | `str`   | `"1233_1234_1235_1236"`                 | Underscore-separated ports for parallel vLLM endpoints      |
| `--temperature`          | `float` | `0.0`                                   | Base sampling temperature                                   |
| `--n`                    | `int`   | `1`                                     | Number of responses per prompt (best-of-n)                  |
| `--split`                | `str`   | `"test"`                                | Dataset split to load (`train`, `test`, etc.)               |
| `--proportion`           | `float` | `1.0`                                   | Fraction of the split to run (0–1)                          |
| `--use_feedback`         | flag    | `False`                                 | Enable answer-level feedback                                |
| `--use_process_feedback` | flag    | `False`                                 | Enable full process-based feedback                          |
| `--use_openai`           | flag    | `False`                                 | Route feedback requests through OpenAI o4-mini              |
| `--logprobs`             | `int`   | `None`                                  | Number of log-probs to request                              |
| `--shuffle`              | flag    | `False`                                 | Shuffle MCQ answer choices between iterations               |
| `--binary_hint`          | flag    | `False`                                 | Provide hints of past incorrect choices                     |
| `--in_temp`              | flag    | `False`                                 | Increase temperature each round                             |
| `--best_of_n`            | flag    | `False`                                 | Enable “best-of-n” repeated sampling per round              |
| `--iterations`           | `int`   | `10`                                    | Total number of refinement rounds                           |

The code supports four different kinds of feedback: Binary feedback, Answer only feedback, Process Solution feedback, and Openai model feedback. The first, second, and the last mode
corresponds to the Binary Correctness Feedback, Self-Generated Reflective Feedback, and Strong-Model Reflective Feedback. 

**Example Usage**:

**Binary Correctness Feedback**: python openai_async_process.py --dataset gpqa  --agent_model meta-llama/Llama-3.3-70B-Instruct --base_url http://c007 --ports 1233 --write_file gpqa_log.jsonl --iterations 10 --proportion 1 --logprobs 1

**Self-Generated Reflective Feedback**: python openai_async_process.py --dataset gpqa  --agent_model meta-llama/Llama-3.3-70B-Instruct --base_url http://c007 --ports 1233 --write_file gpqa_log.jsonl --iterations 10 --proportion 1 --logprobs 1 --use_feedback --use_process_feedback

**Strong-Model Reflective Feedback**: python openai_async_process.py --dataset gpqa  --agent_model meta-llama/Llama-3.3-70B-Instruct --base_url http://c007 --ports 1233 --write_file gpqa_log.jsonl --iterations 10 --proportion 1 --logprobs 1 --use_feedback --use_process_feedback --use_openai
