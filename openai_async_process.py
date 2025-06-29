import asyncio
import json
import os
import random
import re
import sys
import time
from argparse import ArgumentParser

import aiohttp
import datasets
import numpy as np
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import (
    call_openai_feedback,
    call_vllm_server,
    call_vllm_server_batched,
    generate_best_of_n_response,
    generate_question,
    get_dataset_key,
    get_demonstrations,
    get_feedback_messages,
    get_normalized_answer,
    get_normalized_prediction,
    get_normalized_predictions,
    get_previous,
    get_process_answer,
    is_equivalent,
    mask_feedback_answers,
    setup_datalist,
    shuffle_mcq_choices
)

sys.setrecursionlimit(5000)

base_url = ['http://c004']
ports = [1233, 1234, 1235, 1236]
gsm8k_datalist = None
math_datalist = None
iterations = 10 # revise back later
use_feedback = False
shuffle = False
use_process_feedback = False
binary_hint = False
np.random.seed(14)
logprobs = None
letter_to_index = {chr(ord('A') + i): i for i in range(10)}
fluctuate_temp = False
feedback_temp = 0.0
best_of_n = False
openai_feedback = False
# num_print = 0

async def get_response(data, pbar: tqdm, agent_model: str, dataset: str, tokenizer=None, temperature=0.0, n=1, round=0):
    previous = get_previous(dataset, data, round=round) # extract and reformat question
    # remove all cases of "answer it again and then add it at last."
    agent_temp = temperature # we default to 0.0
    if fluctuate_temp:
        # each iteration the agent_temp is updated by round * 0.1 at most to 0.9 at the last response
        
        agent_temp = round * 0.15 # more agressively increase temp
        if agent_temp >= 1.5:
            agent_temp = 1.5
            
    # print("the current temp is: ", agent_temp)
    if round != 0:
        cleaned = previous.replace("Please answer the question again.", "").strip()
        # in the best setting, we do not append the question, or it provide minimal advantages
        # but if we need to switch the answer, we need to append the question.
        if shuffle:
            previous = cleaned + " Please answer the question.\n"
        elif binary_hint:
            # print("added!")
            all_atp = " ".join(f"({a})" for a in data["all_attempts"])
            length_of_choices = 4 if dataset in ["mmlu", "gpqa"] else 10 # 10 for mmlu_pro
            choice_avia = " ".join(
                f"({chr(65 + i)})" for i in range(length_of_choices)
                if chr(65 + i) not in data['all_attempts'] or chr(65 + i) == "X"
            )
            previous = cleaned + "\nYou have previously tried the following answers: " + "[" + all_atp + "]. If (X) exists in your previous answers, it means you didn't provide the answer for some attempts. " + "Now you may need to consider answer choices different from your previous attempts to get to the correct answer, which are: " + "[" + choice_avia + "]."
        else: # base case
            # base 2 is add question
            # base 1 is not add question
            previous = cleaned + "\nQuestion: " + data["original_question"] + " Please answer the question again.\n"
        # this is needed for the shuffle version for reference. / needed for binary_hint for update previous choices. / baseline 2
        # data[get_dataset_key(dataset)] = previous # update after we done this for each iteration, check if this may help. normally we only append at the end
    # print("\nprevious:", previous)
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
    if round == 0 or not best_of_n:
        agent_response, agent_response_probs = await call_vllm_server(agent_model, new_messages, agent_temp, n, tokenizer, base_url, ports, type="answer", dataset=dataset, round=round, logprobs=logprobs)
        print("not using best of n temp: ", agent_temp)
    elif round != 0 and best_of_n:
        agent_response, agent_response_probs = await generate_best_of_n_response(
            agent_model, new_messages, agent_temp, n, tokenizer, base_url, ports,
            dataset, round, logprobs, data
        )
            
                
    response_list = []
    response_list.append(agent_response)
    normalized_prediction_list = get_normalized_predictions(dataset, response_list)
    feedback = ""
    check_leakage = ""
    original_question_combined = ""
    # append the original question and choice options for furture usage.
    # =====================================================================
    if round == 0: # store this at the first iteration and get it out in other rounds
        origin_question = data[get_dataset_key(dataset)]
        marker_prev_answer = "You are given the full history of your previous attempts and feedback provided."
        if marker_prev_answer in origin_question:
            question_part = origin_question.split(marker_prev_answer)[0]
        else:
            question_part = origin_question
        # for dataset without choices this is fine
        original_question_combined = question_part

        # get the choices for multiplication questions
        if dataset == "mmlu":
            choices = "\nChoices: " + "\n".join([f"({chr(ord('A') + i)}) {option}" for i, option in enumerate(data['choices'])])
        elif dataset == "mmlu_pro" or dataset == "gpqa":
            choices = "\nChoices: " + "\n".join([f"({chr(ord('A') + i)}) {option}" for i, option in enumerate(data['options'])])

        if dataset == "mmlu" or dataset == "mmlu_pro" or dataset == "gpqa":
            original_question_combined = question_part + choices
        # print(original_question_combined)
    else:
        original_question_combined = data["original_question"]
    # =======================================================================
    # check if we are answering the question correctly at top
    question_correct = await is_equivalent(dataset, {"normalized_prediction": normalized_prediction_list, "normalized_answer": get_normalized_answer(dataset, data)}, data)
    print("Are the answer and prediction eq? ", question_correct)
    
    if use_feedback:
        # empty or is incorrect
        if len(normalized_prediction_list) == 0 or not question_correct:
            
            # Generate feedback messages based on dataset and configuration
            feedback_messages = get_feedback_messages(
                dataset, original_question_combined, response_list, data, 
                round, use_process_feedback, shuffle
            )     
            # print("feedback_message: ", feedback_messages)
            if openai_feedback:
                print("Using OpenAI gpt-4o-mini model for feedback...")
                feedback_text, feedback_summary, feedback_usage = await call_openai_feedback(feedback_messages)
                feedback = (feedback_text, None)
                data["feedback_summary"] = feedback_summary
                data["feedback_usage"] = feedback_usage

                
            else:
                print("Normal Model!")
                feedback = await call_vllm_server(agent_model, feedback_messages, temperature, n, tokenizer, base_url, ports, type="feedback", dataset=dataset, round=round)

            # Apply dataset-specific answer masking
            feedback = mask_feedback_answers(dataset, feedback, data)
            
    dataset_key = get_dataset_key(dataset) # use this to ensure consistency
    d = {
        dataset_key: generate_question(dataset, data, round=round), # TODO: since we always have a question field, shall we unify the key for question by using "question" instead of others?
        "normalized_answer": get_normalized_answer(dataset, data),
        "normalized_prediction": normalized_prediction_list,
        "full_response": response_list,
        "feedback": feedback,
        "response_probs": agent_response_probs,
        "original_question": original_question_combined,
        "is_correct": question_correct
    }
    # pbar.update(1)
    return d


def apply_async(data_list, agent_model, dataset, tokenizer, temperature, n):
    result_overall, leftover_problems = [[] for _ in range(iterations)], []
    for iters in range(iterations):
        tqdm.write(f"\niteration: {iters}") # record 
        pbar = tqdm(total=len(data_list))
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
        tasks = [loop.create_task(get_response(data, pbar, agent_model, dataset, tokenizer, temperature, n, iters)) for data in data_list]
        result = loop.run_until_complete(asyncio.gather(*tasks))
        data_list_temp = []
        leftover_problems = []
        # iter_output_path = f"gpqa_qwen235B_iter_{iters}_all.jsonl" # need to revise back
        # f_iter = open(iter_output_path, "w")  # open file for incremental logging
        for j in range(len(data_list)):
            # print("The length of datalist is: ", len(data_list))
            item = result[j]
            # if len(item["normalized_prediction"]) >= 1 and item["normalized_answer"] == item["normalized_prediction"][0]:
            # we append other fields back to the item
            for key in data_list[j]: # why different?
                if key not in item:
                    item[key] = data_list[j][key]
            if len(item["normalized_prediction"]) >= 1 and item["is_correct"] is True:
                # item["is_correct"] = True
                result_overall[iters].append(item)
                # f_iter.write(json.dumps(item) + "\n")  # log immediately
            else:
                # item["is_correct"] = False
                # append all answers it has tried for binary hint:
                # if binary_hint: # now for each new answer, we append this
                if "all_attempts" not in item:
                    item["all_attempts"] = []
                # we need to ensure it is not None
                if len(item["normalized_prediction"]) >= 1:
                    if item["normalized_prediction"][0] not in item["all_attempts"]:
                        item["all_attempts"].append(item["normalized_prediction"][0])
                else:
                    # otherwise, we append None to the dataset
                    item["all_attempts"].append("None")

                result_overall[iters].append(item)
                temp = item # revised
                if use_feedback:
                    # we update temp at here datalist[j] -> item since item is the real result
                    # we use this to append the whole history correctly
                    if iters < 1:
                        
                        temp[get_dataset_key(dataset)] = (item[get_dataset_key(dataset)]
                                                        + "\nYou are given the full history of your previous attempts and feedback provided for each attempt.\n"
                                                        + "History:\n"
                                                        + f"\nAttempt at (iteration {iters+1}) and the corresponding feedback:\n"
                                                        + f"\nAnswer:\n{item['full_response'][0]}"
                                                        + "\nYour answer was incorrect."
                                                        + f"\nHere is the feedback:\n{item['feedback'][0]}"
                                                        + "\nPlease answer the question again. "
                                                    )
                        # Shuffle answer choices if enabled
                        if shuffle:
                            temp = shuffle_mcq_choices(dataset, temp, letter_to_index)
                    else:
                        temp[get_dataset_key(dataset)] = (item[get_dataset_key(dataset)]
                                                        + f"\n\nAttempt at (iteration {iters+1}) and the corresponding feedback:\n"
                                                        + f"\nAnswer:\n{item['full_response'][0]}"
                                                        + "\nYour answer was incorrect."
                                                        + f"\nHere is the feedback:\n{item['feedback'][0]}"
                                                        + "\nPlease answer the question again. "
                                                    )     
                        if shuffle:
                            temp = shuffle_mcq_choices(dataset, temp, letter_to_index)                 
                else: # binary feedback
                    if iters < 1:
                        temp[get_dataset_key(dataset)] = (item[get_dataset_key(dataset)] + 
                                                          "\n\nYou are given the full history of your previous attempts and the feedback provided for each attempt.\n"
                                                        + "History:\n"
                                                        + f"\nAttempt at (iteration {iters+1}) and the corresponding feedback:\n"
                                                        + f"\nAnswer:\n{item['full_response'][0]}\n\n"
                                                        + "Feedback: Your answer was incorrect. Please answer the question again\n"
                                                    )
                        # Shuffle answer choices if enabled for binary feedback
                        if shuffle and (not use_feedback) and (not use_process_feedback):
                            temp = shuffle_mcq_choices(dataset, temp, letter_to_index)
                    else: # for iterations after the 1st
                        temp[get_dataset_key(dataset)] = (item[get_dataset_key(dataset)] + 
                                                          f"\nAttempt at (iteration {iters+1}) and the corresponding feedback:\n"
                                                        + f"\nAnswer:\n{item['full_response'][0]}\n\n"
                                                        + "Feedback: Your answer was incorrect. Please answer the question again.\n" #  considering all feedbacks provided
                                                    )
                        # Shuffle answer choices if enabled for binary feedback
                        if shuffle and (not use_feedback) and (not use_process_feedback):
                            temp = shuffle_mcq_choices(dataset, temp, letter_to_index)

                data_list_temp.append(temp)
                # f_iter.write(json.dumps(temp) + "\n")  # log incorrect sample immediately
                # print("\ntemp: ", temp) # sanitycheck 
               
        # f_iter.close()  # CLOSE HERE
        data_list = data_list_temp
        # print("\ndata_list: ", data_list)
        leftover_problems = data_list
        # print(f"\nlength of results over all at iteration {iters}", len(result_overall[iters]))
        # print(f"left over datalist at iteration{iters} ", len(leftover_problems))
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
    parser.add_argument("--use_openai", action="store_true", default=False, help="Use feedback from openai model")
    parser.add_argument("--logprobs", type=int, default=None, help="Logprobs to use for the model")
    parser.add_argument("--shuffle", action="store_true", help="If we need to shuffle the answer choice for multiplication question.")
    parser.add_argument("--binary_hint", action="store_true", help="if we provided previously selected answer choices")
    parser.add_argument("--in_temp", action="store_true", help="whether to increase temperature at each iteration")
    parser.add_argument("--best_of_n", action="store_true", help="whether to generate n different generations per round.")
    
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
    shuffle = args.shuffle
    logprobs = args.logprobs
    binary_hint = args.binary_hint
    fluctuate_temp = args.in_temp
    best_of_n = args.best_of_n
    openai_feedback = args.use_openai
    if agent_model != "meta-llama/Llama-4-Scout-17B-16E-Instruct":
        data_list = setup_datalist(args.dataset, mode=split, random_choice=True)
    else:
        data_list = setup_datalist(args.dataset, mode=split, random_choice=True)
    data_list = data_list[:int(len(data_list) * float(args.proportion))]
    tokenizer = AutoTokenizer.from_pretrained(agent_model)
    chunks = [data_list[x:x+750] for x in range(0, len(data_list), 750)]
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
            for data_item in result[i]:
                data_item["iteration"] = i
                write_file.write(json.dumps(data_item) + '\n')
        # for d in leftover_problems:
        #     d["iteration"] = iterations
        #     write_file.write(json.dumps(d) + '\n')
    write_file.close()
    # print the results that's the sum of the accuracies
    print("Accuracies: ", [round(sum([accuracies[j] for j in range(i + 1)]) * 100 / len(data_list), 3) for i in range(iterations)])
    # print("Accuracies: ", [round(accuracies[i] * 100 / len(data_list), 1) for i in range(iterations)])
    print("Total TIME: ", time.time() - start_time)

