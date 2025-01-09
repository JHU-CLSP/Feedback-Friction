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
from utils import setup_datalist, get_previous, get_demonstrations, get_normalized_answer, get_normalized_prediction, get_dataset_key, call_vllm_server, get_normalized_predictions, generate_question, get_process_answer, is_equivalent, check_if_ground_truth_exists, check_if_ground_truth_exists_mcq
from database import RedisCache


# Connect to your local Redis
cache = None# RedisCache(host='localhost', port=6379, db=0) # try other db=1 on 6379 use 0 for default
base_url = ['http://c004']
ports = [1233, 1234, 1235, 1236]
gsm8k_datalist = None
math_datalist = None
iterations = 10 # revise back later
use_feedback = False
use_process_feedback = False
np.random.seed(14)


async def get_response(data, pbar: tqdm, agent_model: str, dataset: str, tokenizer=None, temperature=0.0, n=1, round=0):
    previous = get_previous(dataset, data) # extract and reformat question
    prediction_list, response_list = [], []
    normalized_answer_list, normalized_prediction_list = [], []
    if dataset == "mmlu_pro":
        new_messages = get_demonstrations(dataset, data['category']).copy()
    elif dataset == "mmlu":
        new_messages = get_demonstrations(dataset, data['subject']).copy()
    else:
        new_messages = get_demonstrations(dataset, category=None).copy()
    new_messages.append({
        "role": "user",
        "content": previous # ask the new question
    })
    
    agent_response = await call_vllm_server(agent_model, new_messages, temperature, n, tokenizer, base_url, ports, cache, type="answer", dataset=dataset, round=round)

    response_list = []
    response_list.append(agent_response)
    normalized_prediction_list = get_normalized_predictions(dataset, response_list)
    feedback = ""
    if use_feedback:
        if len(normalized_prediction_list) == 0 or not is_equivalent(dataset, {"normalized_prediction": normalized_prediction_list, "normalized_answer": get_normalized_answer(dataset, data)}, data):
            if not use_process_feedback:
                feedback_messages = [{"role": "user", "content": "There is a previous mistake on answering this question. Question: " + data[get_dataset_key(dataset)] + "\nAnswer: " + response_list[0] + "\nThe correct final answer should be: " + get_normalized_answer(dataset, data) + "\nPlease give me feedback on which step is wrong or how to get to the correct answer without directly giving up the correct answer: "}]
            else:
                # also provide ground-truth answer trajectory
                feedback_messages = [{"role": "user", "content": "There is a previous mistake on answering this question. Question: " + data[get_dataset_key(dataset)] + "\nAnswer: " + response_list[0] + "\nThe correct final answer should be: " + get_normalized_answer(dataset, data) + "\nThe correct solution that arrives at correct final answer is: " + get_process_answer(dataset, data) + "\nPlease give me feedback on which step is wrong or how to get to the correct answer without directly giving out the correct final answer: "}]
            feedback = await call_vllm_server(agent_model, feedback_messages, temperature, n, tokenizer, base_url, ports, cache, type="feedback", dataset=dataset, round=round)
            # feedback = mask_answer_in_string(feedback, get_normalized_answer(dataset, data))
            # enhance inference time
            if dataset != "mmlu" and dataset != "mmlu_pro" and dataset != "gpqa":
                if check_if_ground_truth_exists(feedback, get_normalized_answer(dataset, data)):
                    feedback_messages = [{"role": "user", "content": "There is a previous mistake on answering this question. Question: " + data[get_dataset_key(dataset)] + "\nAnswer: " + response_list[0] + "\nThe correct final answer should be: " + get_normalized_answer(dataset, data) + "\nThe correct solution that arrives at correct final answer is: " + get_process_answer(dataset, data) + "\nPlease give me feedback on which step is wrong or how to get to the correct answer without DIRECTLY PROVIDING THE CORRECT FINAL ANSWER: "}]
                    feedback = await call_vllm_server(agent_model, feedback_messages, temperature, n, tokenizer, base_url, ports, cache, type="feedback", dataset=dataset, round=round)
            else:
                if check_if_ground_truth_exists_mcq(feedback, get_normalized_answer(dataset, data)):
                    feedback_messages = [{"role": "user", "content": "There is a previous mistake on answering this question. Question: " + data[get_dataset_key(dataset)] + "\nAnswer: " + response_list[0] + "\nThe correct final answer should be: " + get_normalized_answer(dataset, data) + "\nThe correct solution that arrives at correct final answer is: " + get_process_answer(dataset, data) + "\nPlease give me feedback on which step is wrong or how to get to the correct answer without DIRECTLY PROVIDING THE CORRECT FINAL ANSWER: "}]
                    feedback = await call_vllm_server(agent_model, feedback_messages, temperature, n, tokenizer, base_url, ports, cache, type="feedback", dataset=dataset, round=round)
  
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
    # pbar = tqdm(total=10)
    for i in range(iterations):
        tqdm.write(f"iteration: {i}") # record 
        pbar = tqdm(total=len(data_list))
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
        tasks = [loop.create_task(get_response(data, pbar, agent_model, dataset, tokenizer, temperature, n, i)) for data in data_list]
        result = loop.run_until_complete(asyncio.gather(*tasks))
        data_list_temp = []
        for j in range(len(data_list)):
            item = result[j]
            # if len(item["normalized_prediction"]) >= 1 and item["normalized_answer"] == item["normalized_prediction"][0]:
            if len(item["normalized_prediction"]) >= 1 and is_equivalent(dataset, item, data_list[j]):
                result_overall[i].append(item)
            else:
                temp = data_list[j]
                if use_feedback:
                    # print(item["full_response"])
                    temp[get_dataset_key(dataset)] = data_list[j][get_dataset_key(dataset)] + "\n\nPrevious Answer: " + item["full_response"][0] + "\n\n" + "Your previous answer is incorrect.\n" + "Here is some feedback: " + item["feedback"] + "\nAnswer the question again.\n"
                else:
                    temp[get_dataset_key(dataset)] = data_list[j][get_dataset_key(dataset)] + "\n\nPrevious Answer: " + item["full_response"][0] + "\n\n" + "Your previous answer is incorrect. Answer the question again.\n"
                data_list_temp.append(temp)
        data_list = data_list_temp
        leftover_problems = data_list
        tqdm.write(f"correct overall: {len(result_overall[i])}") # record 
        # print("leftover in this epoch:" + str(len(leftover_problems[i])))
        tqdm.write(f"leftover in this epoch: {len(data_list)}")
        #pbar.update(i)
        loop.close()
    
    return result_overall, leftover_problems


if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math", help="dataset to test with")
    parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Agent model to use for generating responses")
    parser.add_argument("--write_file", type=str, default="output_arc.jsonl", help="File to write the output to")
    parser.add_argument("--base_url", type=str, default="http://c004", help="Base URL to use for the agent server")
    parser.add_argument("--ports", type=str, default="1233_1234_1235_1236", help="Base URL to use for the agent server")
    parser.add_argument("--temperature", type=str, default="0.0", help="Base temperature to use for inference")
    parser.add_argument("--n", type=str, default="1", help="Base best_of n to use for inference")
    parser.add_argument("--split", type=str, default="test", help="Split to use for the dataset")
    parser.add_argument("--proportion", type=str, default="1", help="Proportion of the dataset to use")
    parser.add_argument("--use_feedback", action="store_true", help="Use feedback for the model")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--use_process_feedback", action="store_true", help="Use process feedback")
    
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
    iterations = args.iterations
    use_process_feedback = args.use_process_feedback
    data_list = setup_datalist(args.dataset, mode=split)
    data_list = data_list[:int(len(data_list) * float(args.proportion))]
    tokenizer = AutoTokenizer.from_pretrained(agent_model)
    chunks = [data_list[x:x+500] for x in range(0, len(data_list), 500)]
    print(len(chunks))
    accuracies = [0 for _ in range(iterations)]
    cnt = 0
    # running and post-training statistics collection
    for chunk in chunks:
        cnt += 1
        print("epoch number: " + str(cnt))
        result, leftover_problems = apply_async(chunk, agent_model, dataset, tokenizer, temperature, n)
        # print(time.time() - start_time) # time until this epoch
        for i in range(iterations):
            accuracies[i] += len(result[i])
        for d in leftover_problems:
            write_file.write(json.dumps(d) + '\n')
    write_file.close()
    # print the results that's the sum of the accuracies
    print("Accuracies: ", [round(sum([accuracies[j] for j in range(i + 1)]) * 100 / len(data_list), 1) for i in range(iterations)])
    # print("Accuracies: ", [round(accuracies[i] * 100 / len(data_list), 1) for i in range(iterations)])
    print("Total TIME: ", time.time() - start_time)
    
    output_file = "gpqa31_405B_outcome.txt"  # Specify the output file name

    # Calculate the accuracies
    accuracies_list = [round(sum([accuracies[j] for j in range(i + 1)]) * 100 / len(data_list), 1) for i in range(iterations)]

    # Calculate total time
    total_time = time.time() - start_time

    # Write results to the file
    with open(output_file, "w") as file:  # Use "a" if you want to append to the file
        file.write(f"Accuracies: {accuracies_list}\n")
        file.write(f"Total TIME: {total_time:.2f} seconds\n")


