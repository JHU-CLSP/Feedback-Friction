from transformers import AutoTokenizer
import datasets
import json
import time
import numpy as np
import re
import random
import re
import random
import asyncio
import aiohttp
from tqdm import tqdm
from argparse import ArgumentParser
from utils import setup_datalist, get_previous, get_demonstrations, get_normalized_answer, get_normalized_prediction, get_dataset_key, call_vllm_server, get_normalized_predictions, generate_question, get_process_answer, is_equivalent, check_if_ground_truth_exists
from database import RedisCache


# Connect to your local Redis
cache = None #RedisCache(host='localhost', port=6379)
base_url = ['http://c002']
ports = [1233, 1234, 1235, 1236]
gsm8k_datalist = None
math_datalist = None
iterations = 1
use_feedback = False
use_process_feedback = False
np.random.seed(14)
category_map = {
    "impossible to solve": 0,
    "too complicated": 0,
    "feedback is wrong": 0,
    "model is not following the feedback": 0,
    "style or formalization issue": 0,
    "unknown": 0
}


async def get_response(data, pbar: tqdm, agent_model: str, dataset: str, tokenizer=None, temperature=0.0, n=1, round=0):
    question = generate_question(dataset, data)
    process_answer = get_process_answer(dataset, data)
    question = "Your previous answer is".join(question.split("Your previous answer is")[0: 5])
    
    new_messages = [{
        "role": "system",
        "content": "You are an error categorizer specialized in analyzing why Language Learning Models (LLMs) fail to self-improve when solving problems. When provided with an LLM's prediction trajectory and the feedback it receives, you will categorize the errors into one of six categories:\n\n1. Problem is Impossible to Solve\n   - The problem itself is fundamentally flawed\n   - External tools are required (e.g., calculator for complex calculations, search engine for obscure facts)\n\n2. Problem is Too Complicated\n   - The problem exceeds the model's knowledge scope\n   - Example: A level 5 math problem beyond the model's training\n\n3. Feedback is Wrong\n   - The feedback generator model provides incorrect guidance\n   - The feedback fails to identify actual mistakes in the model's response\n   - The feedback is too vague or generic to be helpful\n   - You can identify these issues by comparing the feedback against the provided ground truth answer\n\n4. Model is Not Following Feedback\n   - The model fails to incorporate or properly implement the given feedback\n   - This includes cases where the feedback model provides the correct answer, but the model still cannot generate it\n\n5. Style or Formalization Issue\n   - The answer is logically correct but has problems with representation, such as:\n   - Mathematical notation: e.g., writing boxed{[-2, 7]} vs boxed{x ∈ [-2,7]} when either is acceptable\n   - Formatting preferences: using different but equivalent representations\n   - Presentation style: verbose vs. concise expressions of the same concept\n   - Formal notation variations that don't affect mathematical correctness\n   - Different but logically equivalent ways of expressing the same solution\n   - Minor stylistic issues that don't impact the correctness of the solution\n\n6. Unknown\n   - The error cannot be clearly categorized into any of the above categories\n\nWhen providing your analysis, you should:\n1. Clearly explain your reasoning for choosing a particular category\n2. Support your categorization with specific examples from the provided trajectory\n3. End your response with:\n\n\"The error is: [category]\" where category is one of:\n- impossible to solve\n- too complicated\n- feedback is wrong\n- model is not following the feedback\n- style or formalization issue\n- unknown"
    }]
    
    new_messages.append({
        "role": "user",
        "content": "The question and the interaction between the agent model and the feedback model is: " + question + "\n\nThe process answer is: " + process_answer
    })
    new_messages.append({
        "role": "user",
        "content": "Please categorize the error."
    })
    
    agent_response = await call_vllm_server(agent_model, new_messages, temperature, n, tokenizer, base_url, ports, cache, type="category", dataset=dataset, round=round)

    response_list = []
    response_list.append(agent_response)

    try:
        category = agent_response.split("The error is: ")[1].strip()
    except:
        category = "unknown"
    if category not in category_map:
        if "impossible to solve" in category:
            category = "impossible to solve"
        elif "too complicated" in category:
            category = "too complicated"
        elif "feedback is wrong" in category:
            category = "feedback is wrong"
        elif "model is not following the feedback" in category:
            category = "model is not following the feedback"
        elif "style or formalization issue" in category:
            category = "style or formalization issue"
        else:
            category = "unknown"

    d = {
        "question": generate_question(dataset, data),
        "full_response": response_list,
        "category": category,
        "answer": process_answer
    }
    pbar.update(1)
    return d

def apply_async(data_list, agent_model, dataset, tokenizer, temperature, n):
    result_overall, leftover_problems = [[] for _ in range(iterations)], None
    for i in range(iterations):
        pbar = tqdm(total=len(data_list))
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
        tasks = [loop.create_task(get_response(data, pbar, agent_model, dataset, tokenizer, temperature, n, i)) for data in data_list]
        result = loop.run_until_complete(asyncio.gather(*tasks))
        result_overall[i] = result
        loop.close()
    
    return result_overall, leftover_problems


if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math", help="dataset to test with")
    parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help="Agent model to use for generating responses")
    parser.add_argument("--write_file", type=str, default="output_arc.jsonl", help="File to write the output to")
    parser.add_argument("--base_url", type=str, default="http://c004", help="Base URL to use for the agent server")
    parser.add_argument("--ports", type=str, default="1233_1234_1235_1236", help="Base URL to use for the agent server")
    parser.add_argument("--temperature", type=str, default="0.0", help="Base temperature to use for inference")
    parser.add_argument("--n", type=str, default="1", help="Base best_of n to use for inference")
    parser.add_argument("--split", type=str, default="test", help="Split to use for the dataset")
    parser.add_argument("--file", type=str, default="result.json", help="The file of output")
    
    # prepare the arguments
    args = parser.parse_args()
    agent_model = args.agent_model
    base_url = [args.base_url]
    ports = [int(item) for item in args.ports.split("_")]
    dataset = args.dataset
    write_file = open(args.write_file, 'w')
    temperature = float(args.temperature)
    n = int(args.n)
    split = args.split
    data_list = []
    for line in open(args.file):
        data_list.append(json.loads(line))
    tokenizer = AutoTokenizer.from_pretrained(agent_model)
    chunks = [data_list[x:x+500] for x in range(0, len(data_list), 500)]
    accuracies = [0 for _ in range(iterations)]
    
    # running and post-training statistics collection
    for chunk in chunks:
        result, leftover_problems = apply_async(chunk, agent_model, dataset, tokenizer, temperature, n)
        for i in range(iterations):
            for j in range(len(chunk)):
                item = result[i][j]
                category = item["category"]
                category_map[category] += 1
                write_file.write(json.dumps(item) + "\n")
    
    write_file.close()
    # print the results that's the sum of the accuracies
    print("Categories: ", category_map)
    print("Categories percentage: ", {k: round(v * 100 / len(data_list), 1) for k, v in category_map.items()})
    # print("Accuracies: ", [round(accuracies[i] * 100 / len(data_list), 1) for i in range(iterations)])
    print("Total TIME: ", time.time() - start_time)