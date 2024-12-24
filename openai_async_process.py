from transformers import AutoTokenizer
import datasets
import json
import time
import numpy as np
import re
import random
import re
import random
from tqdm import tqdm
import asyncio
import aiohttp
from argparse import ArgumentParser
from utils import remove_boxed, last_boxed_only_string, strip_string, remove_right_units, normalize_answer, extract_and_convert_number_real

base_url = ['http://c003']
ports = [1233, 1234, 1235, 1236]
# ports = [1233]
gsm8k_datalist = None
math_datalist = None
iterations = 10

def get_url():
    return random.choice(base_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'

messages_start_rationale = [
    {
        "role": "system",
        "content": "Your task is to give me chain of thought answers for the reasoning problem"
    },
]

list_of_end_prompts = ['the answer is', 'The correct answer is', 'The answer is', 'The answer is indeed', 'the correct answer is indeed']

def setup_datalist(dataset_name, mode="test"):
    if dataset_name == "arc":
        ds = datasets.load_dataset("allenai/ai2_arc", 'ARC-Challenge')
        data_list = list(ds['test'])
        global arc_datalist
        arc_datalist = list(ds['train'])
        if mode == "test":
            return data_list
        elif mode == "train":
            return arc_datalist
    elif dataset_name == "gsm8k":
        ds = datasets.load_dataset("gsm8k", 'main')
        data_list = list(ds['test'])
        global gsm8k_datalist
        gsm8k_datalist = list(ds['train'])
        if mode == "test":
            return data_list
        elif mode == "train":
            return gsm8k_datalist
    elif dataset_name == "math":
        ds = datasets.load_dataset("lighteval/MATH", 'all')
        data_list = list(ds['test'])
        global math_datalist
        math_datalist = list(ds['train'])
        # s = "/scratch/dkhasha1/djiang21/new_project/model_training/finetuning_data/math_train_easy.jsonl"
        # math_datalist = []
        # for line in open(s, 'r'):
        #     t = json.loads(line)
        #     if "\\boxed" in t['solution']:
        #         math_datalist.append(t)
        # return math_datalist
        if mode == "test":
            return data_list
        elif mode == "train":
            return math_datalist
    elif dataset_name == "trivia_qa":
        ds = datasets.load_dataset("mandarjoshi/trivia_qa", 'rc.nocontext')
        data_list = list(ds['validation'])
        global triviaqa_datalist
        triviaqa_datalist = list(ds['train'])
        if mode == "test":
            return data_list
        elif mode == "train":
            return triviaqa_datalist

def get_previous(dataset_name, data):
    if dataset_name == "arc":
        # previous = "Question: " + data['question'] + '\nChoices: '
        # for i in range(len(data['choices']['text'])):
        #     previous += chr(ord('A') + i) + " - " + data['choices']['text'][i] + " "
        # previous = previous[:-1] + '\nAnswer:'
        previous = data['question'] + "\n\nChoices: " + '\n'.join(data["choices"]["text"])
        return previous
    elif dataset_name == "gsm8k":
        previous = "Question: " + data['question'] + '\nAnswer: '
        return previous
    elif dataset_name == "math":
        # return "Question: " + data['problem'] + "\nAnswer:"
        return data['problem'] + "\nAnswer:"
    elif dataset_name == "trivia_qa":
        return data['question']

def get_messages(dataset_name):
    if dataset_name == "gsm8k":
        messages_gsm8k = []
        rand_list_from_train = np.random.choice(gsm8k_datalist, 9, replace=False)
        for data in rand_list_from_train:
            l = []
            d = {"role": "user", "content": "Question: " + data['question'] + "\nAnswer:"}
            l.append(d)
            data['answer'] = data['answer'].replace("####", "The answer is:")
            answers = data['answer'].split("\n")
            # for answer in answers:
            #     if answer == "":
            #         continue
            #     if not answer.endswith("."):
            #         answer = answer + "."
            #     l.append({"role": "assistant", "content": answer})
            l.append({"role": "assistant", "content": data['answer']})
            messages_gsm8k.extend(l)
        return messages_gsm8k
    elif dataset_name == "math":
        messages_math = [{"role": "system", "content": "You are a smart assistant that solves math problems. If you think you're ready to output the answer, you can wrap your answer with \\boxed{}. Please follow this format metrically"}]
        rand_list_from_train = np.random.choice(math_datalist, 5, replace=False)
        for data in rand_list_from_train:
            l = []
            d = {"role": "user", "content": "Question: " + data['problem'] + "\nAnswer:"}
            l.append(d)
            # answers = data['solution'].split(". ")
            # for answer in answers:
            #     if answer == "":
            #         continue
            #     l.append({"role": "assistant", "content": answer})
            l.append({"role": "assistant", "content": data['solution']})
            messages_math.extend(l)
        return messages_math
    elif dataset_name == "arc":
        messages_arc = [{"role": "system", "content": "You are a smart assistant that solves reasoning problems. If you think you're ready to output the answer, you can just output an answer in A, B, C or D. Please just output one character and follow this format metrically"}]
        rand_list_from_train = np.random.choice(arc_datalist, 8, replace=False)
        for data in rand_list_from_train:
            l = []
            d = {"role": "user", "content": data['question'] + "\n\nChoices: " + '\n'.join(data["choices"]["text"])}
            l.append(d)
            l.append({"role": "assistant", "content": data['answerKey']})
            messages_arc.extend(l)
        return messages_arc
    elif dataset_name == "trivia_qa":
        messages_triviaqa = [{"role": "system", "content": "You are a smart assistant that solves trivia questions. If you think you're ready to output the answer, you can just output an answer."}]
        rand_list_from_train = np.random.choice(triviaqa_datalist, 8, replace=False)
        for data in rand_list_from_train:
            l = []
            d = {"role": "user", "content": data['question']}
            l.append(d)
            l.append({"role": "assistant", "content": data['answer']["normalized_aliases"][0]})
            messages_triviaqa.extend(l)
        return messages_triviaqa

np.random.seed(14)

def get_normalized_answer(dataset_name, data):
    if dataset_name == "arc" or dataset_name == "ecqa":
        return data['answerKey']
    elif dataset_name == "gsm8k":
        return extract_and_convert_number_real(data['answer'].split("####")[1].strip())
    elif dataset_name == "math":
        solution = data['solution']
        res = remove_boxed(last_boxed_only_string(solution))
        try:
            res = strip_string(res)
        except:
            pass
        return res
    elif dataset_name == "trivia_qa":
        return data['answer']["normalized_value"]

def get_dataset_key(dataset_name):
    if dataset_name == "arc" or dataset_name == "ecqa" or dataset_name == "gsm8k" or dataset_name == "mmlu_pro":
        return "question"
    elif dataset_name == "math":
        return "problem"
    elif dataset_name == "proofwriter":
        return "context"
    elif dataset_name == "trivia_qa":
        return "question"

def get_normalized_prediction(dataset_name, prediction):
    if dataset_name == "arc" or dataset_name == "ecqa" or dataset_name == "proofwriter":
        normalized_prediction = prediction.strip().replace(": ", "")
        return normalized_prediction
    elif dataset_name == "gsm8k":
        return extract_and_convert_number_real(prediction)
    elif dataset_name == "math":
        res = remove_boxed(last_boxed_only_string(prediction))
        try:
            res = strip_string(res)
        except:
            pass
        return res
    elif dataset_name == "trivia_qa":
        return normalize_answer(prediction)
        
async def get_response(data, pbar: tqdm, agent_model: str, dataset: str, tokenizer=None, temperature=0.0, n=1):
    previous = get_previous(dataset, data)
    prediction_list, response_list = [], []
    normalized_answer_list, normalized_prediction_list = [], []
    
    new_messages = get_messages(dataset).copy()
    # new_messages = []
    new_messages.append({
        "role": "user",
        "content": previous
    })
    # print(new_messages)
    url = get_url()
    content = {
        "model": agent_model,
        "messages": new_messages,
        "max_tokens": 1000,
        "temperature": temperature,
        "stop_token_ids": [128001, 128009],
        "best_of": n,
        "n": n,
        "logprobs": 1,
        "seed": 14,
        "chat_template": tokenizer.chat_template
    }
    headers = {
        "Content-Type": "application/json"
    }
    session_timeout = aiohttp.ClientTimeout(total=60000,sock_connect=6000,sock_read=6000)

    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        async with session.post(url, headers=headers, json=content) as agent_response:
            try:
                agent_response.raise_for_status()
                agent_response = await agent_response.json()
            except Exception as e:
                print(e)
                print("Error in calling remote agent server")

    response_list = []
    prob_list = []
    try:
        for output in agent_response['choices']:
            response = output['message']['content']
            prob = output['logprobs']['content']
            total_prob = sum([word['logprob'] for word in prob])
            # print(total_prob / len(prob))
            response_list.append(response)
            prob_list.append(total_prob / len(prob))
    except:
        response = ""
        response_list.append(response)
    # max_idx = np.argmax(np.array(prob_list))
    # response = response_list[max_idx]

    if dataset == "gsm8k":
        for term in list_of_end_prompts:
            try:
                if term in response:
                    prediction = response.split(term)[1].replace('\n', ' ').strip()
                    normalized_prediction = get_normalized_prediction(dataset, prediction)
                    if normalized_prediction == "":
                        pass
                    else:
                        normalized_prediction_list.append(normalized_prediction)
                        break
            except:
                print("Error")
    elif dataset == "math":
        for response in response_list:
            try:
                prediction = response
                normalized_prediction = get_normalized_prediction(dataset, prediction)
                normalized_prediction_list.append(normalized_prediction)
            except:
                print("Error")
    elif dataset == "arc":
        for response in response_list:
            try:
                prediction = response
                normalized_prediction = prediction
                normalized_prediction_list.append(normalized_prediction)
            except:
                print("Error")
    elif dataset == "trivia_qa":
        for response in response_list:
            try:
                prediction = response
                normalized_prediction = get_normalized_prediction(dataset, prediction)
                normalized_prediction_list.append(normalized_prediction)
            except:
                print("Error")
    
    feedback = ""
    if len(normalized_prediction_list) != 0 and normalized_prediction_list[0] != get_normalized_answer(dataset, data):
        url = get_url()
        feedback_messages = [{"role": "user", "content": "There is a previous mistake on answering this question. Question: " + data[get_dataset_key(dataset)] + "\nAnswer: " + response_list[0] + "\nThe correct final answer should be: " + get_normalized_answer(dataset, data) + "\nPlease give me feedback on which step is wrong or how to get to the correct answer without giving up the correct answer: "}]
        content = {
            "model": agent_model,
            "messages": feedback_messages,
            "max_tokens": 1000,
            "temperature": temperature,
            "stop_token_ids": [128001, 128009],
            "best_of": n,
            "n": n,
            "logprobs": 1,
            "seed": 14,
            "chat_template": tokenizer.chat_template
        }
        headers = {
            "Content-Type": "application/json"
        }
        session_timeout = aiohttp.ClientTimeout(total=60000,sock_connect=6000,sock_read=6000)

        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            async with session.post(url, headers=headers, json=content) as agent_response:
                try:
                    agent_response.raise_for_status()
                    agent_response = await agent_response.json()
                except Exception as e:
                    print(e)
                    print("Error in calling remote agent server")
        try:
            for output in agent_response['choices']:
                response = output['message']['content']
                # print(total_prob / len(prob))
                feedback = response
        except:
            feedback = ""
        

    if dataset == "arc":
        question = data[get_dataset_key(dataset)] + "\n\nChoices: " + '\n'.join(data["choices"]["text"])
    elif dataset == "math" or dataset == "trivia_qa" or dataset == "gsm8k":
        question = data[get_dataset_key(dataset)]
    d = {
        "question": question,
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
                temp[get_dataset_key(dataset)] = data_list[j][get_dataset_key(dataset)] + "\n\nPrevious Answer: " + item["full_response"][0] + "\n\n" + "Your previous answer is incorrect.\n" + "Here is some feedback: " + item["feedback"] + "\nAnswer the question again."
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
    
    
    args = parser.parse_args()
    agent_model = args.agent_model
    base_url = [args.base_url]
    ports = [int(item) for item in args.ports.split("_")]

    dataset = args.dataset
    write_file = open(args.write_file, 'w')
    temperature = float(args.temperature)
    n = int(args.n)
    split = args.split

    data_list = setup_datalist(args.dataset, mode=split)
    if args.proportion != "1":
        data_list = data_list[:int(len(data_list) * float(args.proportion))]
    tokenizer = AutoTokenizer.from_pretrained(agent_model)

    chunks = [data_list[x:x+500] for x in range(0, len(data_list), 500)]
    accuracies = [0 for _ in range(iterations)]
    for chunk in chunks:
        result, leftover_problems = apply_async(chunk, agent_model, dataset, tokenizer, temperature, n)
        for i in range(iterations):
            accuracies[i] += len(result[i])
        for d in leftover_problems:
            write_file.write(json.dumps(d) + '\n')
    write_file.close()
    print("Accuracies: ", [round(accuracies[i] * 100 / len(data_list), 1) for i in range(iterations)])
    print("Total TIME: ", time.time() - start_time)
