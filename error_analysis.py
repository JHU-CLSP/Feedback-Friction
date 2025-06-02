import asyncio
import json
import os
import time
from argparse import ArgumentParser

import numpy as np
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import call_vllm_server, generate_question, get_process_answer, get_dataset_key

# vLLM server configuration
base_url = ['http://c002']
ports = [1233, 1234, 1235, 1236]
iterations = 1
np.random.seed(14)

# Error categories counter
tag_map = {
    "impossible to solve": 0,
    "too complicated": 0,
    "feedback is wrong": 0,
    "model is not following the feedback": 0,
    "style or formalization issue": 0,
    "unknown": 0
}

# OpenAI reasoning client setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def call_openai_judge(messages):
    """Call the OpenAI reasoning-enabled model (o4-mini) to categorize errors using Responses API.
    Returns: (response_text, summary, usage)
    """
    try:
        response = client.responses.create(
            model="o4-mini",
            reasoning={"effort": "medium", "summary": "auto"},
            input=messages,
            max_output_tokens=1500
        )

        # Extract response text
        if response.status == "incomplete" and getattr(response, 'incomplete_details', None) and response.incomplete_details.reason == "max_output_tokens":
            text = response.output_text or ""
        else:
            text = response.output_text

        # Correctly extract summary from response.reasoning
        summary = None
        if hasattr(response, 'reasoning'):
            reasoning = response.reasoning
            if isinstance(reasoning, dict):
                summary = reasoning.get("summary")
            elif hasattr(reasoning, "summary"):
                summary = reasoning.summary

        # Extract and serialize usage
        raw_usage = getattr(response, 'usage', None)
        usage = raw_usage.dict() if raw_usage and hasattr(raw_usage, 'dict') else raw_usage

        return text.strip(), summary, usage

    except Exception as e:
        return f"OpenAI API Error: {e}", None, {}





async def get_response(data, pbar: tqdm, agent_model: str, dataset: str,
                       tokenizer=None, temperature=0.0, n=1, round=0):
    """Fetch and categorize an answer trajectory for iteration 9 incorrect items."""
    question = data[get_dataset_key(dataset)]
    process_answer = get_process_answer(dataset, data)

    # Truncate after the 5th occurrence of "Here is the feedback"
    segments = question.split("\nHere is the feedback:")
    if len(segments) > 5:
        question = "\nHere is the feedback:".join(segments[:5]) + "\n[Further feedback truncated]"

    # Construct prompt messages
    messages = [
        {"role": "system", "content": (
            "You are an error categorizer specialized in analyzing why Language Learning Models (LLMs) "
            "fail to self-improve when solving problems. When provided with an LLM's prediction trajectory "
            "and the feedback it receives, you will categorize the errors into one of six categories:\n\n"
            "1. Problem is Impossible to Solve\n   - The problem itself is fundamentally flawed\n   - External tools are required (e.g., calculator for complex calculations, search engine for obscure facts)\n\n"
            "2. Problem is Too Complicated\n   - The problem exceeds the model's knowledge scope\n   - Example: A level 5 math problem beyond the model's training\n\n"
            "3. Feedback is Wrong\n   - The feedback generator model provides incorrect guidance\n   - The feedback fails to identify actual mistakes in the model's response\n   - The feedback is too vague or generic to be helpful\n\n"
            "4. Model is Not Following Feedback\n   - The model fails to incorporate feedback properly\n\n"
            "5. Style or Formalization Issue\n   - Logical correctness but formatting or notation problems\n\n"
            "6. Unknown\n   - Cannot categorize the failure into above categories\n\n"
            "End your response with exactly: 'The error is: [category]' where category is one of:\n- impossible to solve\n- too complicated\n- feedback is wrong\n- model is not following the feedback\n- style or formalization issue\n- unknown"
        )}
    ]
    user_content = (
        f"The question and the interaction between the agent model and the feedback model is: {question}\n\n"
        f"The process answer is: {process_answer}\n\n"
        "Please categorize the error. End your response with:\n'The error is: [category]'"
    )
    messages.append({"role": "user", "content": user_content})

    if "openai" in agent_model.lower():
        response_text, summary, usage = await call_openai_judge(messages)
        response_list = [response_text]
    else:
        agent_response = await call_vllm_server(
            agent_model,
            messages,
            temperature,
            n,
            tokenizer,
            base_url,
            ports,
            cache=None,
            type="category",
            dataset=dataset,
            round=round
        )
        response_text = agent_response[0]
        response_list = agent_response
        summary = None
        usage = None

    # Normalize category
    try:
        category = response_text.split("The error is:")[1].strip().lower()
    except:
        category = "unknown"
    if category not in tag_map:
        category = next((k for k in tag_map if k in category), "unknown")

    result = {
        "question": generate_question(dataset, data),
        "full_response": response_list,
        "summary": summary,
        "usage": usage,
        "category": category,
        "answer": process_answer
    }
    pbar.update(1)
    return result

def apply_async(data_list, agent_model, dataset, tokenizer, temperature, n):
    results = [[] for _ in range(iterations)]
    for i in range(iterations):
        pbar = tqdm(total=len(data_list))
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
        tasks = [loop.create_task(get_response(item, pbar, agent_model, dataset, tokenizer, temperature, n, i)) for item in data_list]
        batch = loop.run_until_complete(asyncio.gather(*tasks))
        results[i] = batch
        loop.close()
    return results

if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--write_file", type=str, default="output_arc.jsonl")
    parser.add_argument("--base_url", type=str, default="http://h15")
    parser.add_argument("--ports", type=str, default="1233_1234_1235_1236")
    parser.add_argument("--file", type=str, default="result.json")
    args = parser.parse_args()

    agent_model = args.agent_model
    base_url = [args.base_url]
    ports = [int(p) for p in args.ports.split("_")]
    dataset = args.dataset

    data_list = []
    with open(args.file) as f:
        for line in f:
            item = json.loads(line)
            if item.get("iteration", 0) == 9 and not item.get("is_correct", False):
                data_list.append(item)
    if agent_model != "openai":
        tokenizer = AutoTokenizer.from_pretrained(agent_model)
    else:
        tokenizer = None
        
    data_list = data_list[:2]
    chunks = [data_list[i:i+750] for i in range(0, len(data_list), 750)]

    with open(args.write_file, 'w') as out_f:
        for chunk in chunks:
            categorized = apply_async(chunk, agent_model, dataset, tokenizer, 0.0, 1)
            for batch in categorized:
                for res in batch:
                    tag_map[res["category"]] += 1
                    out_f.write(json.dumps(res) + "\n")

    total = len(data_list) or 1
    print("Categories:", tag_map)
    print("Percentages:", {k: round(v*100/total, 1) for k, v in tag_map.items()})
    print("Total Time:", time.time() - start_time)

    with open("qwen_categorization_log_scout.txt", "a") as log_f:
        log_f.write(f"Categories: {tag_map}\n")
        log_f.write(f"Percentages: {{k: v*100/{total} for k, v in tag_map.items()}}\n")
        log_f.write(f"Total Time: {time.time() - start_time}\n\n")