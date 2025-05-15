# Self-InPrefect
Repository for our paper RIGID THINKING: LLMs Struggle to Fully Incorporate External Feedback

The **main file** for our experiments is openai_async_process.py which has the following command:

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

We offer other functionalities such as --use_best_of_n which is the part of rejection sampling in our code. --in_temp which increases the temperature per iteration --binary_hint refers to provide models of its previous incorrect choices and ask it not to do that again --shuffle refers to shuffing the positions of answer choices in the question for MCQ questions 

To use the --use_openai you should fill in your openai api key in the AsyncOpenAI Client with your api key. 

Currently supporting datasets:
mmlu, mmlu_pro, gpqa, math (refers to math-500), custom_simple (refers to 5 digits multiplication questions), hex (refers to 5 digits hexidecimal multiplication questions), aime_2024, trivia_qa, pop_qa. Note gsm8k and gsmsymbolic are not used anymore in our evaluation and are deprecated in the setting. 

Here is a brief summary of the functionalities of other files:
manual_hints_5d.py: providing hints used for 5 digits multiplications
start_multiple_server...sh: files for starting vllm server to host model on clusters
check_dis_new.py: script for checking the "in-dsirtibutionness" of the model's output by generating 100 outputs per question
error_analysis.py: script used for classifying whether model follows feedback based on openai model's judgement. Also need api key input.

Above are all the main files contributing to our experiment setup.
