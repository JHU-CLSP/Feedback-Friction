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
from utils import setup_datalist, get_previous, get_messages, get_normalized_answer, get_normalized_prediction, get_dataset_key, call_vllm_server, extract_predictions, generate_question, get_process_answer

base_url = ['http://c003']
ports = [1233, 1234, 1235, 1236]
gsm8k_datalist = None
math_datalist = None
iterations = 10
use_feedback = False
np.random.seed(14)


async def get_response(data, pbar: tqdm, agent_model: str, dataset: str, tokenizer=None, temperature=0.0, n=1):
    previous = get_previous(dataset, data)
    prediction_list, response_list = [], []
    normalized_answer_list, normalized_prediction_list = [], []
    
    new_messages = get_messages(dataset).copy()
    new_messages.append({
        "role": "user",
        "content": previous
    })
    
    agent_response = await call_vllm_server(agent_model, new_messages, temperature, n, tokenizer, base_url, ports)

    response_list = []
    try:
        for output in agent_response['choices']:
            response = output['message']['content']
            response_list.append(response)
    except:
        response = ""
        response_list.append(response)

    normalized_prediction_list = extract_predictions(dataset, response_list)
    
    feedback = ""
    if use_feedback:
        if len(normalized_prediction_list) != 0 and normalized_prediction_list[0] != get_normalized_answer(dataset, data):
            # feedback_messages = [{"role": "user", "content": "There is a previous mistake on answering this question. Question: " + data[get_dataset_key(dataset)] + "\nAnswer: " + response_list[0] + "\nThe correct final answer should be: " + get_normalized_answer(dataset, data) + "\nPlease give me feedback on which step is wrong or how to get to the correct answer without directly giving up the correct answer: "}]
            # also provide ground-truth answer trajectory
            feedback_messages = [{"role": "user", "content": "There is a previous mistake on answering this question. Question: " + data[get_dataset_key(dataset)] + "\nAnswer: " + response_list[0] + "\nThe correct final answer should be: " + get_normalized_answer(dataset, data) + "\nThe correct solution that arrives at correct final answer is: " + get_process_answer(dataset, data) + "\nPlease give me feedback on which step is wrong or how to get to the correct answer without directly giving up the correct answer: "}]
            agent_response = await call_vllm_server(agent_model, feedback_messages, temperature, n, tokenizer, base_url, ports)
            try:
                for output in agent_response['choices']:
                    response = output['message']['content']
                    feedback = response
            except:
                feedback = ""
    
    d = {
        "question": generate_question(dataset, data),
        "normalized_answer": get_normalized_answer(dataset, data),
        "normalized_prediction": normalized_prediction_list,
        "full_response": response_list,
        "feedback": feedback
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
        tasks = [loop.create_task(get_response(data, pbar, agent_model, dataset, tokenizer, temperature, n)) for data in data_list]
        result = loop.run_until_complete(asyncio.gather(*tasks))
        data_list_temp = []
        for j in range(len(data_list)):
            item = result[j]
            if len(item["normalized_prediction"]) >= 1 and item["normalized_answer"] == item["normalized_prediction"][0]:
                result_overall[i].append(item)
            else:
                temp = data_list[j]
                if use_feedback:
                    temp[get_dataset_key(dataset)] = data_list[j][get_dataset_key(dataset)] + "\n\nPrevious Answer: " + item["full_response"][0] + "\n\n" + "Your previous answer is incorrect.\n" + "Here is some feedback: " + item["feedback"] + "\nAnswer the question again."
                else:
                    temp[get_dataset_key(dataset)] = data_list[j][get_dataset_key(dataset)] + "\n\nPrevious Answer: " + item["full_response"][0] + "\n\n" + "Your previous answer is incorrect. Answer the question again."
                data_list_temp.append(temp)
        data_list = data_list_temp
        leftover_problems = data_list
        loop.close()
    
    return result_overall, leftover_problems

if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math", help="dataset to test with")
    parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Agent model to use for generating responses")
    parser.add_argument("--write_file", type=str, default="output_arc.jsonl", help="File to write the output to")
    parser.add_argument("--base_url", type=str, default="http://c015", help="Base URL to use for the agent server")
    parser.add_argument("--ports", type=str, default="1233_1234_1235_1236", help="Base URL to use for the agent server")
    parser.add_argument("--temperature", type=str, default="0.0", help="Base temperature to use for inference")
    parser.add_argument("--n", type=str, default="1", help="Base best_of n to use for inference")
    parser.add_argument("--split", type=str, default="test", help="Split to use for the dataset")
    parser.add_argument("--proportion", type=str, default="1", help="Proportion of the dataset to use")
    parser.add_argument("--use_feedback", action="store_true", help="Use feedback for the model")
    
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
    use_feedback = args.use_feedback
    data_list = setup_datalist(args.dataset, mode=split)
    if args.proportion != "1":
        data_list = data_list[:int(len(data_list) * float(args.proportion))]
    tokenizer = AutoTokenizer.from_pretrained(agent_model)
    chunks = [data_list[x:x+500] for x in range(0, len(data_list), 500)]
    accuracies = [0 for _ in range(iterations)]
    
    # running and post-training statistics collection
    for chunk in chunks:
        result, leftover_problems = apply_async(chunk, agent_model, dataset, tokenizer, temperature, n)
        for i in range(iterations):
            accuracies[i] += len(result[i])
        for d in leftover_problems:
            write_file.write(json.dumps(d) + '\n')
    write_file.close()
    # print the results that's the sum of the accuracies
    print("Accuracies: ", [round(sum([accuracies[j] for j in range(i + 1)]) * 100 / len(data_list), 1) for i in range(iterations)])
    # print("Accuracies: ", [round(accuracies[i] * 100 / len(data_list), 1) for i in range(iterations)])
    print("Total TIME: ", time.time() - start_time)

