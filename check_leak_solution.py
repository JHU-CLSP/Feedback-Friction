from transformers import AutoTokenizer
import json
import time
import numpy as np
import asyncio
from tqdm import tqdm
from argparse import ArgumentParser
from utils import call_vllm_server, generate_question, get_process_answer


iterations = 10 
np.random.seed(14)

async def get_response(data, pbar: tqdm, agent_model: str, dataset: str, tokenizer=None, temperature=0.0, n=1, round=0):
    question = generate_question(dataset, data)
    process_answer = get_process_answer(dataset, data)
    question = question.split("Here is some feedback: ")[-1]  # only check the last iteration
    
    new_messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant which output whether the feedback model leaks the solution. When provided the feedback, the solution, "
                "and the question, you will categorize the feedback into the following category: \n"
                "1. directly leak final solution. (when the model provides feedback that leaks the ground truth) \n"
                "2. leak almost entirely (when the model solves the problem directly without leaving any space for the model "
                "that receives responses to think) \n"
                "3. appropriate feedback. (the feedback is appropriate) \n"
                "4. unknown. You should output one sentence of why you categorize the feedback like that and end with: "
                "'The feedback is: [category].', where [category] is one of the four categories mentioned above."
            ),
        },
        {"role": "user", "content": f"The feedback is: {question}\n\nThe solution is: {process_answer}"},
        {"role": "user", "content": "Please categorize the feedback."},
    ]
    
    agent_response = await call_vllm_server(
        agent_model, new_messages, temperature, n, tokenizer, base_url, ports, cache=None, type="category", dataset=dataset, round=round
    )
    
    response_list = [agent_response]


    category = "none"  # default to "unknown"
    '''
    try:
        # Look for the specific phrase "The feedback is: [category]"
        if "The feedback is: " in agent_response:
            print(agent_response)
            category = agent_response.split("The feedback is: ")[-1].strip().lower()
    except Exception as e:
        print(f"Error parsing category: {e}")
    '''

    valid_categories = ["directly leak final solution", "leak almost entirely", "appropriate feedback", "unknown"]
    if category not in valid_categories:
        if "directly leak final solution" in agent_response[0]:
            category = "directly leak final solution"
        elif "leak almost entirely" in agent_response[0]:
            category = "leak almost entirely"
        elif "appropriate feedback" in agent_response[0]:
            category = "appropriate feedback"
        else:
            category = "unknown"
    
    pbar.update(1)
    return {
        "question": question,
        "full_response": response_list,
        "category": category,
        "answer": process_answer
    }

def apply_async(data_list, agent_model, dataset, tokenizer, temperature, n):
    result_overall = []
    for i in range(1, iterations):
        pbar = tqdm(total=len(data_list[i]))
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
        tasks = [loop.create_task(get_response(data, pbar, agent_model, dataset, tokenizer, temperature, n, i)) for data in data_list[i]]
        result = loop.run_until_complete(asyncio.gather(*tasks))
        result_overall.append(result)
        loop.close()
    return result_overall

if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math", help="Dataset to test with")
    parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Agent model to use for generating responses")
    parser.add_argument("--write_file", type=str, default="output_arc.jsonl", help="File to write the output to")
    parser.add_argument("--base_url", type=str, default="http://c010", help="Base URL to use for the agent server")
    parser.add_argument("--ports", type=str, default="1233_1234_1235_1236", help="Ports to use for the agent server")
    parser.add_argument("--temperature", type=str, default="0.0", help="Temperature for inference")
    parser.add_argument("--n", type=str, default="1", help="Number of responses to generate per request")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--file", type=str, default="result.json", help="Input JSONL file")

    args = parser.parse_args()
    agent_model = args.agent_model
    base_url = [args.base_url]
    ports = [int(item) for item in args.ports.split("_")]
    dataset = args.dataset
    temperature = float(args.temperature)
    n = int(args.n)

    data_list = [[] for _ in range(iterations)]
    with open(args.file, "r") as f:
        for line in f:
            item = json.loads(line)
            number = item['iteration']
            if 0 <= number < iterations:
                data_list[number].append(item)
            else:
                print(f"Skipping item with invalid iteration {number}")

    tokenizer = AutoTokenizer.from_pretrained(agent_model)

    results = apply_async(data_list, agent_model, dataset, tokenizer, temperature, n)

    category_maps = [{} for _ in range(iterations)]
    with open(args.write_file, "w") as write_file:
        for i, bucket_result in enumerate(results):
            for item in bucket_result:
                category = item["category"]
                if category not in category_maps[i]:
                    category_maps[i][category] = 0
                category_maps[i][category] += 1
                write_file.write(json.dumps(item) + "\n")

    for i in range(iterations): # get data for each iteration 
        total_items = sum(category_maps[i].values())
        if total_items > 0:
            print(f"Iteration {i} Categories: {category_maps[i]}")
            print(f"Iteration {i} Percentages: ", {
                k: round(v * 100 / total_items, 1) for k, v in category_maps[i].items()
            })
        else:
            print(f"Iteration {i} is empty.")

    print("Total TIME: ", time.time() - start_time)
