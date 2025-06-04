from transformers import AutoTokenizer
import datasets
import json
import time
import numpy as np
import asyncio
import aiohttp
from tqdm import tqdm
from argparse import ArgumentParser
from utils import (
    setup_datalist,
    get_previous,
    get_demonstrations,
    get_normalized_answer,
    get_normalized_predictions,
    generate_question,
    is_equivalent,
    get_url
)


# Global variables
cache = None
base_url = ['http://c004']
ports = [1233, 1234, 1235, 1236]
gsm8k_datalist = None
math_datalist = None
np.random.seed(14)
logprobs = None

# ---------------------------
# Global concurrency control
# ---------------------------
MAX_CONCURRENT_REQUESTS = 10000  # Adjust if needed
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def call_vllm_server_batched(
    agent_model,
    new_messages,
    temperature,
    tokenizer,
    base_url,
    ports,
    cache,
    type=None,
    dataset=None,
    round=None,
    logprobs=None,
    expected_answer=None,
    n=1
):
    """
    Make a single call to the vLLM server with the given parameters.
    'n' controls how many generations (samples) are requested in one call.
    """
    url = get_url(base_url, ports)
    content = {
        "model": agent_model,
        "messages": new_messages,
        "max_tokens": 2000,
        "temperature": temperature,
        "stop_token_ids": [128001, 128009, tokenizer.eos_token_id],
        "best_of": n,
        "n": n,
        "logprobs": logprobs,
        "seed": 14,
        "chat_template": tokenizer.chat_template
    }
    headers = {"Content-Type": "application/json"}
    session_timeout = aiohttp.ClientTimeout(total=60000, sock_connect=6000, sock_read=6000)

    response_json = None
    max_retries = 1

    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        for attempt in range(max_retries):
            try:
                async with semaphore:  # limit concurrency
                    async with session.post(url, headers=headers, json=content) as resp:
                        # print("Status:", resp.status)
                        # print("Body:", await resp.text())            
                        resp.raise_for_status()
                        response_json = await resp.json()
                        break  # success
            except Exception as e:
                print(f"[Retry {attempt+1}] Error in calling remote agent server: {e}")
                await asyncio.sleep(2 ** attempt)  # exponential backoff

    responses = []
    probs = []

    if response_json is not None:
        for choice in response_json.get('choices', []):
            resp_text = choice.get('message', {}).get('content', "")
            responses.append(resp_text)

            if logprobs is not None:
                logprob_items = choice.get('logprobs', {}).get('content', [])
                if len(logprob_items) > 1:
                    avg_logprob = sum(item["logprob"] for item in logprob_items[:-1]) / (len(logprob_items) - 1)
                else:
                    avg_logprob = None
                probs.append(avg_logprob)
            else:
                probs.append(None)
    else:
        # If the response failed all retries
        responses = ["" for _ in range(n)]
        probs = [None for _ in range(n)]

    return responses, probs

async def get_response(
    data,
    pbar: tqdm,
    agent_model: str,
    dataset: str,
    tokenizer=None,
    temperature=0.3,
    round=0,
    attempts=10,
    generations_per_attempt=10
):
    """
    For a single question, we do 'attempts' calls to the server.
    Each call requests 'generations_per_attempt' generations, so total = attempts * generations_per_attempt.
    Then we collect all responses, see how many match, and store metadata.
    """
    # Prepare the prompt
    previous = get_previous(dataset, data, round=0)
    if dataset == "mmlu_pro":
        new_messages = get_demonstrations(dataset, data['category']).copy()
    elif dataset == "mmlu":
        new_messages = get_demonstrations(dataset, data['subject']).copy()
    else:
        new_messages = get_demonstrations(dataset, category=None).copy()

    new_messages.append({
        "role": "user",
        "content": previous
    })

    expected_answer = get_normalized_answer(dataset, data)

    # We'll accumulate all generations across attempts
    all_responses = []
    all_probs = []

    for i in range(attempts):
        # call the server once, requesting 'generations_per_attempt' outputs
        responses, probs = await call_vllm_server_batched(
            agent_model,
            new_messages,
            temperature,
            tokenizer,
            base_url,
            ports,
            cache,
            type="answer",
            dataset=dataset,
            round=i,
            logprobs=logprobs,
            expected_answer=expected_answer,
            n=generations_per_attempt
        )
        all_responses.extend(responses)
        all_probs.extend(probs)

    # Now we have attempts * generations_per_attempt responses
    normalized_predictions = get_normalized_predictions(dataset, all_responses)

    # Evaluate how many are correct
    correct_attempts = []
    for prediction in normalized_predictions:
        temp_resp = {
            "normalized_prediction": [prediction],
            "normalized_answer": expected_answer
        }
        if dataset != 'math':
            if is_equivalent(dataset, temp_resp, data):
                correct_attempts.append(prediction)
        else:
            temp_resp = {
                "normalized_prediction": [prediction],
                "normalized_answer": expected_answer
            }
            if prediction == expected_answer or is_equivalent(dataset, temp_resp, data):
                correct_attempts.append(prediction)

    is_correct = len(correct_attempts) > 0
    total_gens = attempts * generations_per_attempt
    percentage_correct = (len(correct_attempts) / total_gens) * 100

    result = {
        "question": generate_question(dataset, data),
        "normalized_answer": expected_answer,
        "normalized_prediction": normalized_predictions,
        "full_response": all_responses,
        "response_probs": all_probs,
        "is_correct": is_correct,
        "percentage_correct": percentage_correct,
        "attempts": total_gens
    }

    # pbar.update(1)  # if you want to increment progress bar per question
    return result

async def process_question(data, agent_model, dataset, tokenizer, temperature, attempts, generations_per_attempt, pbar):
    """
    Process a single question by calling get_response.
    """
    response = await get_response(
        data,
        pbar,
        agent_model,
        dataset,
        tokenizer,
        temperature,
        round=0,
        attempts=attempts,
        generations_per_attempt=generations_per_attempt
    )
    return response


def apply_async(data_list, agent_model, dataset, tokenizer, temperature, attempts, generations_per_attempt):
    """
    Runs all questions asynchronously, gathering results.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    pbar = tqdm(total=len(data_list))

    tasks = [
        loop.create_task(
            process_question(
                data,
                agent_model,
                dataset,
                tokenizer,
                temperature,
                attempts,
                generations_per_attempt,
                pbar
            )
        )
        for data in data_list
    ]

    results = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    return results


if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math", help="Dataset to test with")
    parser.add_argument(
        "--agent_model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Agent model to use for generating responses"
    )
    parser.add_argument("--write_file", type=str, default="output_arc.jsonl", help="File to write the output to")
    parser.add_argument("--base_url", type=str, default="http://c004", help="Base URL to use for the agent server")
    parser.add_argument("--ports", type=str, default="1233_1234_1235_1236", help="Ports to use for the agent server")
    parser.add_argument("--temperature", type=str, default="0.3", help="Temperature to use for inference")

    # 'attempts' => how many calls per question
    parser.add_argument("--attempts", type=int, default=10, help="Number of calls per question")
    # 'gens' => how many generations per call
    parser.add_argument("--gens", type=int, default=10, help="Number of generations per call")

    parser.add_argument("--split", type=str, default="test", help="Split to use for the dataset")
    parser.add_argument("--proportion", type=str, default="1", help="Proportion of the dataset to use")
    parser.add_argument("--logprobs", type=int, default=None, help="Logprobs to use for the model")
    parser.add_argument("--use_feedback", action="store_true", help="Use feedback")

    args = parser.parse_args()

    # Setup from args
    agent_model = args.agent_model
    base_url = [args.base_url]
    ports = [int(item) for item in args.ports.split("_")]
    dataset = args.dataset

    write_file = open(args.write_file, 'w')

    temperature = float(args.temperature)
    print("using temperature: ", temperature)
    attempts = args.attempts
    generations_per_attempt = args.gens
    logprobs = args.logprobs
    split = args.split
    data_list = setup_datalist(args.dataset, mode=split)
    data_list = data_list[:int(len(data_list) * float(args.proportion))]

    tokenizer = AutoTokenizer.from_pretrained(agent_model)

    # Partition into chunks of size 100 if desired
    chunks = [data_list[x:x+100] for x in range(0, len(data_list), 100)]
    print(f"Number of chunks: {len(chunks)}")

    all_results = []

    for cnt, chunk in enumerate(chunks, start=1):
        print("Epoch number:", cnt)
        results = apply_async(
            chunk,
            agent_model,
            dataset,
            tokenizer,
            temperature,
            attempts,
            generations_per_attempt
        )
        all_results.extend(results)
        for data in results:
            write_file.write(json.dumps(data) + '\n')

    write_file.close()

    num_questions = len(data_list)
    num_correct = len([item for item in all_results if item["is_correct"]])
    overall_percentage = round(num_correct * 100 / num_questions, 3) if num_questions > 0 else 0
    print("Overall Accuracy:", overall_percentage, "%")
    print("Total TIME:", time.time() - start_time)
