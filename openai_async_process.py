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
from utils import setup_datalist, get_previous, get_demonstrations, get_normalized_answer, get_normalized_prediction, get_dataset_key, call_vllm_server, get_normalized_predictions, generate_question, get_process_answer, is_equivalent, mask_answer_in_string_arith
from database import RedisCache
from manual_hints_5d import provide_multiplication_hints, extract_numbers_and_process_5d, extract_numbers_and_process_6d, extract_numbers_and_process_4d, extract_numbers_and_process_7d
from openai import OpenAI

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
logprobs = None


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
    
    agent_response, agent_response_probs = await call_vllm_server(agent_model, new_messages, temperature, n, tokenizer, base_url, ports, cache, type="answer", dataset=dataset, round=round, logprobs=logprobs)

    response_list = []
    response_list.append(agent_response)
    normalized_prediction_list = get_normalized_predictions(dataset, response_list)
    feedback = ""
    if use_feedback:
        if len(normalized_prediction_list) == 0 or not is_equivalent(dataset, {"normalized_prediction": normalized_prediction_list, "normalized_answer": get_normalized_answer(dataset, data)}, data):
            if not use_process_feedback and dataset != "custom_simple":

                feedback_messages = [{"role": "user", "content": "There is a previous mistake on answering this question. Question: " + data[get_dataset_key(dataset)] + "\nAnswer: " + response_list[0] + "\nThe correct final answer should be: " + get_normalized_answer(dataset, data) + "\nPlease give me feedback on which step is wrong or how to get to the correct answer without directly giving up the correct answer: "}]

            elif dataset == "custom_simple": # feedback for arith questions
                # TODO: for future, I can directly add the feedback for each question into the dataset instead of using the function here
                feedback_messages = [{"role": "user", "content":  "There is a previous mistake on answering this question. Question: " + data[get_dataset_key(dataset)] +  "\nPrevious Answer: " + response_list[0] + "\nThe correct final answer should be: " + get_normalized_answer(dataset, data) + "The correct steps that lead to the final answer is: " + extract_numbers_and_process_7d(str(data[get_dataset_key(dataset)])) +"\nBased on the correct calculation process, please give me feedback on which step was wrong or how to get to the correct answer in detail: "}]
                
            else:
                # also provide ground-truth answer trajectory
                feedback_messages = [{"role": "user", "content": "There is a previous mistake on answering this question. Question: " + data[get_dataset_key(dataset)] + "\nAnswer: " + response_list[0] + "\nThe correct final answer should be: " + get_normalized_answer(dataset, data) + "\nThe correct solution that arrives at correct final answer is: " + get_process_answer(dataset, data) + "\nPlease give me feedback on which step is wrong or how to get to the correct answer without directly giving up the correct answer: "}]
            
            feedback = await call_vllm_server(agent_model, feedback_messages, temperature, n, tokenizer, base_url, ports, cache, type="feedback", dataset=dataset, round=round)

            # enhance inference time
            # TODO: please check if we can delete this completely
            '''
            if dataset != "mmlu" and dataset != "mmlu_pro" and dataset != "gpqa": # revisied to get the feedback
                if check_if_ground_truth_exists(feedback[0], get_normalized_answer(dataset, data)):
                    feedback_messages = [{"role": "user", "content": "There is a previous mistake on answering this question. Question: " + data[get_dataset_key(dataset)] + "\nAnswer: " + response_list[0] + "\nThe correct final answer should be: " + get_normalized_answer(dataset, data) + "\nThe correct solution that arrives at correct final answer is: " + get_process_answer(dataset, data) + "\nPlease give me feedback on which step is wrong or how to get to the correct answer without DIRECTLY PROVIDING THE CORRECT FINAL ANSWER: "}]
                    feedback = await call_vllm_server(agent_model, feedback_messages, temperature, n, tokenizer, base_url, ports, cache, type="feedback", dataset=dataset, round=round)
                    if check_if_ground_truth_exists(feedback[0], get_normalized_answer(dataset, data)): # mask answer
                        feedback = mask_answer_in_string(feedback[0], get_normalized_answer(dataset, data))
            else:
                if check_if_ground_truth_exists_mcq(feedback[0], get_normalized_answer(dataset, data)):
                    feedback_messages = [{"role": "user", "content": "There is a previous mistake on answering this question. Question: " + data[get_dataset_key(dataset)] + "\nAnswer: " + response_list[0] + "\nThe correct final answer should be: " + get_normalized_answer(dataset, data) + "\nThe correct solution that arrives at correct final answer is: " + get_process_answer(dataset, data) + "\nPlease give me feedback on which step is wrong or how to get to the correct answer WITHOUT DIRECTLY PROVIDING THE CORRECT FINAL ANSWER: "}]
                    feedback = await call_vllm_server(agent_model, feedback_messages, temperature, n, tokenizer, base_url, ports, cache, type="feedback", dataset=dataset, round=round)
            '''
            # do it for MCQs
            if dataset == "mmlu" or dataset == "mmlu_pro" or dataset == "gpqa":
                # added to extract the original question for good formatting, don't need all questions
                origin_question = data[get_dataset_key(dataset)]
                marker_prev_answer = "\n\nPrevious Answer:"
                if marker_prev_answer in origin_question:
                    question_part = origin_question.split(marker_prev_answer)[0]
                else:
                    question_part = origin_question
                
                # get the choices
                if dataset == "mmlu":
                    choices = "\nChoices: " + '\n'.join(data['choices'])
                else:
                    choices = "\nChoices: " + '\n'.join(data['options'])
                    
                if not use_process_feedback: # should have different feedback messages
                    revise_message = [{"role": "user", "content": "There is a feedback for the question: " + question_part + " " + choices + " which has the ground truth " + get_normalized_answer(dataset, data) + "Check if the feedback leaks the ground truth. If so, remove the ground truth from the feedback and provide the feedback again. "}]
                else:
                    revise_message = [{"role": "user", "content": "There is a feedback for the question: " + question_part + " " + choices + " which has the ground truth " + get_normalized_answer(dataset, data) + "\nThe correct solution that arrives at correct final answer is: " + get_process_answer(dataset, data) + "Check if the feedback leaks the ground truth. If so, remove the ground truth from the feedback and provide the feedback again. "}]
                
                # new feedback after revision
                feedback =  await call_vllm_server(agent_model, revise_message, temperature, n, tokenizer, base_url, ports, cache, type="feedback", dataset=dataset, round=round)
            
            if dataset == "custom_simple":
                feedback = (mask_answer_in_string_arith(feedback[0], get_normalized_answer(dataset, data)), None) # direct musk

    d = {
        "question": generate_question(dataset, data), # TODO: since we always have a question field, shall we unify the key for question by using "question" instead of others?
        "normalized_answer": get_normalized_answer(dataset, data),
        "normalized_prediction": normalized_prediction_list,
        "full_response": response_list,
        "feedback": feedback,
        "response_probs": agent_response_probs
    }
    # pbar.update(1)
    return d

def apply_async(data_list, agent_model, dataset, tokenizer, temperature, n):
    result_overall, leftover_problems = [[] for _ in range(iterations)], []
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
        leftover_problems = []
        for j in range(len(data_list)):
            item = result[j]
            # if len(item["normalized_prediction"]) >= 1 and item["normalized_answer"] == item["normalized_prediction"][0]:
            for key in data_list[j]:
                if key not in item:
                    item[key] = data_list[j][key]
            if len(item["normalized_prediction"]) >= 1 and is_equivalent(dataset, item, data_list[j]):
                item["is_correct"] = True
                result_overall[i].append(item)
            else:
                item["is_correct"] = False
                result_overall[i].append(item)
                temp = data_list[j]
                if use_feedback:
                    # print(item["full_response"])
                    temp[get_dataset_key(dataset)] = data_list[j][get_dataset_key(dataset)] + "\n\nPrevious Answer: " + item["full_response"][0] + "\n\n" + "Your previous answer is incorrect.\n" + "Here is some feedback: " + item["feedback"][0] + "\nAnswer the question.\n" # revised:  again by considering all feedbacks before # need to revise back!
                else:
                    temp[get_dataset_key(dataset)] = data_list[j][get_dataset_key(dataset)] + "\n\nPrevious Answer: " + item["full_response"][0] + "\n\n" + "Your previous answer is incorrect. Answer the question again.\n"
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
    parser.add_argument("--base_url", type=str, default="http://c004", help="Base URL to use for the agent server")
    parser.add_argument("--ports", type=str, default="1233_1234_1235_1236", help="Base URL to use for the agent server")
    parser.add_argument("--temperature", type=str, default="0.0", help="Base temperature to use for inference")
    parser.add_argument("--n", type=str, default="1", help="Base best_of n to use for inference")
    parser.add_argument("--split", type=str, default="test", help="Split to use for the dataset")
    parser.add_argument("--proportion", type=str, default="1", help="Proportion of the dataset to use")
    parser.add_argument("--use_feedback", action="store_true", help="Use feedback for the model")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--use_process_feedback", action="store_true", help="Use process feedback")
    parser.add_argument("--logprobs", type=int, default=None, help="Logprobs to use for the model")
    
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
    logprobs = args.logprobs
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
            accuracies[i] += len([item for item in result[i] if item["is_correct"]])
            for data in result[i]:
                data["iteration"] = i
                write_file.write(json.dumps(data) + '\n')
        # for d in leftover_problems:
        #     d["iteration"] = iterations
        #     write_file.write(json.dumps(d) + '\n')
    write_file.close()
    # print the results that's the sum of the accuracies
    print("Accuracies: ", [round(sum([accuracies[j] for j in range(i + 1)]) * 100 / len(data_list), 3) for i in range(iterations)])
    # print("Accuracies: ", [round(accuracies[i] * 100 / len(data_list), 1) for i in range(iterations)])
    print("Total TIME: ", time.time() - start_time)

