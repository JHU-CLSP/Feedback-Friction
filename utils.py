import re
import string
import random
import datasets
import asyncio
import aiohttp
import numpy as np
import json
from datetime import datetime
from dataset_specific_utils import normalize_final_answer, remove_boxed, last_boxed_only_string, is_equiv, extract_and_convert_number_real, strip_string, normalize_answer, gsm8k_list_of_end_prompts, strip_string_mult
random.seed(42)
np.random.seed(42)
MAX_CONCURRENT_REQUESTS = 1000  # Adjust if needed
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
from openai import AsyncOpenAI

# create initials
client = AsyncOpenAI(api_key="key")
dataset_files = {
        "custom_simple": "digits_buckets/multiplication_questions_6d.jsonl",
        "hex": "hex5d.jsonl"
}

async def call_openai_judge(messages, max_retries=1, wait_seconds=2):
    """
    Call the OpenAI gpt-4.1-mini model as a judge. Prompts must be constructed
    so the model answers only "YES" or "NO". If the response can't be parsed,
    defaults to "NO".
    Returns:
      judge_response (str): "YES" or "NO"
      summary (str|None): any summary returned (unused here)
      usage (dict): token usage
    """
    for attempt in range(max_retries):
        try:
            response = await client.responses.create(
                model="gpt-4.1-mini",
                input=messages,
                max_output_tokens=16
            )

            # normalize text
            text = (response.output_text or "").strip().upper()
            usage = getattr(response, "usage", {}) or {}

            # any mention of YES → accept
            if "YES" in text:
                return "YES", None, usage

            # everything else → NO
            return "NO", None, usage

        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(wait_seconds)
            else:
                # on repeated failure, default to NO
                return "NO", None, {}

# 1) A little sync wrapper around your async judge
async def judge_pop_qa(question: str, prediction: str, possible_answers: list[str]) -> bool:
    """
    Returns True if GPT judge says YES, False otherwise.
    """
    possible_answers_str = ', '.join(possible_answers)
    messages = [
        {"role": "system", "content":
            "You are an answer validator.  "
            "Given a question, a model's answer, and the list of possible correct answers, "
            "respond with ONLY YES if the answer is correct, or NO otherwise."
        },
        {"role": "user", "content":
            f"Question: {question}\n"
            f"Model answer: {prediction}\n"
            f"Possible correct answers: {possible_answers_str}\n"
            "Is the model answer one of the correct answers? Reply YES or NO."
        }
    ]

    verdict, _, _ = await call_openai_judge(messages)
    verdict = verdict.strip().upper()
    return "YES" in verdict, verdict

def get_url(base_url, ports):
    return random.choice(base_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'

# load the costumsized dataset
def load_dataset_local(dataset_name):
    
    if dataset_name in dataset_files:
        file_path = dataset_files[dataset_name]
        try:
            with open(file_path, 'r') as file:
                return [json.loads(line) for line in file]
        except FileNotFoundError:
            print(f"Dataset {dataset_name} not found. Please ensure {file_path} exists. You may need to create the dataset by running generate_arith_questions.py")
            return []
        except json.JSONDecodeError:
            print(f"Error decoding JSON in {file_path}. Please check the file format.")
            return []

def setup_datalist(dataset_name, mode="test", random_choice = False):
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
    elif dataset_name == "gsm8k_symbolic":
        ds = datasets.load_dataset("apple/GSM-Symbolic", name="main")
        data_list = list(ds['test'])
        global gsm8k_symbolic_datalist
        ds_gsm8k = datasets.load_dataset("gsm8k", 'main')
        gsm8k_symbolic_datalist = list(ds_gsm8k['test'])
        if mode == "test":
            return data_list
        elif mode == "train":
            return gsm8k_symbolic_datalist
    elif dataset_name == "math":
        ds = datasets.load_dataset("HuggingFaceH4/MATH-500", name='default')
        data_list = list(ds['test'])
        # global math_datalist
        # math_datalist = list(ds['train'])
        # s = "/scratch/dkhasha1/djiang21/new_project/model_training/finetuning_data/math_train_easy.jsonl"
        # math_datalist = []
        # for line in open(s, 'r'):
        #     t = json.loads(line)
        #     if "\\boxed" in t['solution']:
        #         math_datalist.append(t)
        # return math_datalist
        if mode == "test":
            return data_list
        # elif mode == "train":
        #     return math_datalist
        # note math-500 dataset does not have the training set
    elif dataset_name == "trivia_qa":
        ds = datasets.load_dataset("mandarjoshi/trivia_qa", 'rc.wikipedia.nocontext')
        data_list = list(ds['validation'])
        global triviaqa_datalist
        triviaqa_datalist = list(ds['train'])

        if mode == "test":
            if random_choice:
                return random.sample(data_list, len(data_list) // 10)
            else:
                return data_list
        elif mode == "train":
            return triviaqa_datalist
        
    if dataset_name == "pop_qa":
        # 1) Load the full test split (we’ll treat it as our entire dataset)
        ds = datasets.load_dataset("akariasai/PopQA")
        all_data = list(ds["test"])

        # 2) Parse possible_answers (JSON strings → Python lists)
        for ex in all_data:
            raw = ex.get("possible_answers")
            if isinstance(raw, str):
                try:
                    ex["possible_answers"] = json.loads(raw)
                except json.JSONDecodeError:
                    ex["possible_answers"] = []

        # 3) Cache the “train” split globally (same as all_data)
        global popqa_datalist
        popqa_datalist = all_data
        sample_size = max(1, len(all_data) // 10)
        popqa_test_set = random.sample(popqa_datalist, k=sample_size)
        
        if mode == "test":
            # 4) Return a random sample of 1/10th of the dataset
            return popqa_test_set

        elif mode == "train":
            # 5) Return the full dataset
            popqa_datalist = [ex for ex in popqa_datalist if ex not in popqa_test_set]
            return popqa_datalist

    elif dataset_name == "mmlu":  # adding mmlu dataset
        ds = datasets.load_dataset("cais/mmlu", "all")
        data_list = list(ds['test'])
        global mmlu_datalist_all

        mmlu_datalist_all = {}
        mmlu_datalist_all_ls = list(ds['validation'])
        for data in mmlu_datalist_all_ls:
            if data['subject'] not in mmlu_datalist_all:
                mmlu_datalist_all[data['subject']] = []
            mmlu_datalist_all[data['subject']].append(data)

        if mode == "test":
            if random_choice:
                return random.sample(data_list, len(data_list) // 10)
            else:
                return data_list
        elif mode == "train":
            return mmlu_datalist_all
        
    elif dataset_name == "mmlu_pro":
        ds = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
        data_list = list(ds['test'])
        global mmlu_datalist

        mmlu_datalist = {}
        validation_list = list(ds['validation'])
        for data in validation_list:
            if data['category'] not in mmlu_datalist:
                mmlu_datalist[data['category']] = []
            mmlu_datalist[data['category']].append(data)

        if mode == "test":
            if random_choice:
                return random.sample(data_list, len(data_list) // 10)
            else:
                return data_list
        elif mode == "train":
            return mmlu_datalist

    elif dataset_name == "aime_2024":
        ds = datasets.load_dataset("Maxwell-Jia/AIME_2024")
        data_list = list(ds['train'])
        if mode == "test":
            return data_list
        elif mode == "train":
            print("This Dataset Does Not Have a Valid Training Set!")
            return None
    
    # TODO: double check
    elif dataset_name == "gpqa": # revised for adding the 'Explanation' field into the reformatted dataset
        original_dataset = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond")
        formatted_dataset = datasets.load_dataset("jeggers/gpqa_formatted", 'diamond')
        original_data_list = list(original_dataset['train'])
        formatted_data_list = list(formatted_dataset['train'])
        # TODO: check this update of field from Question to question since otherwise may cause problem
        # update the key from "Question" to "question" in both datasets
        for item in original_data_list:
            item['question'] = item.pop('Question', None)

        for item in formatted_data_list:
            item['question'] = item.pop('Question', None)
        original_mapping = {item['question']: item.get('Explanation', None) for item in original_data_list}
        # Add the "Explanation" field to the formatted dataset
        for entry in formatted_data_list:
            entry_id = entry['question']
            if entry_id in original_mapping:
                entry['Explanation'] = original_mapping[entry_id]
        # ds = datasets.load_dataset("jeggers/gpqa_formatted", 'diamond')
        # data_list = list(ds['train'])
        global gpqa_datalist
        gpqa_datalist = list(datasets.load_dataset("jeggers/gpqa_formatted", 'main')['train'])
        return formatted_data_list
    
    # added for multiplication questions
    elif dataset_name == "custom_simple":
        custom_list = load_dataset_local("custom_simple")
        sample_size = max(1, len(custom_list) // 10)
        return random.sample(custom_list, k=sample_size)

    elif dataset_name == "hex":
        # 1000 questions in total
        hex_list = load_dataset_local("hex")

        return hex_list


# TODO: do we need to reprompt the model of the original question at the end again? see example below:
def get_previous(dataset_name, data, round):
    if dataset_name == "arc":
        # previous = "Question: " + data['question'] + '\nChoices: '
        # for i in range(len(data['choices']['text'])):
        #     previous += chr(ord('A') + i) + " - " + data['choices']['text'][i] + " "
        # previous = previous[:-1] + '\nAnswer:'
        previous = data['question'] + "\n\nChoices: " + '\n'.join(data["choices"]["text"])
        return previous
    elif dataset_name == "gsm8k" or dataset_name == "gsm8k_symbolic":
        previous = "Question: " + data['question'] + '\nAnswer: '
        return previous
    elif dataset_name == "math":
        # return "Question: " + data['problem'] + "\nAnswer:"
        return data['problem']
    elif dataset_name == "trivia_qa" or dataset_name == "pop_qa":
        return data['question']
    elif dataset_name == "aime_2024":
        return data['Problem']
    elif dataset_name == "mmlu":
        # problematic for future iterations
        origin_question = data["question"]
        # marker_prev_answer = "You are given the full history of your previous attempts and feedback provided."
        if round >= 1: # if previously exists then we append full question
            question_part = origin_question # + "\nQuestion: " + origin_question.split(marker_prev_answer)[0]
            previous = question_part
        else: # if this is the first attempt
            question_part = origin_question
            previous = "Question: " + question_part + "\nChoices:\n"
            for i in range(len(data['choices'])):
                previous += f"({chr(ord('A') + i)}) " + data['choices'][i] + "\n"
            previous += "Answer:"
        return previous

    elif dataset_name == "mmlu_pro":
        # problematic for future iterations
        origin_question = data["question"]
        # marker_prev_answer = "You are given the full history of your previous attempts and feedback provided."
        if round >= 1: # if previously exists then we append full question
            question_part = origin_question # + "\nQuestion: " + origin_question.split(marker_prev_answer)[0]
            previous = question_part
        else: # if this is the first attempt
            question_part = origin_question
            previous = "Question: " + question_part + "\nChoices:\n"
            for i in range(len(data['options'])):
                previous += f"({chr(ord('A') + i)}) " + data['options'][i] + "\n"
            previous += "Answer:"
        return previous
    
    elif dataset_name == "gpqa": #revised
        # problematic for future iterations
        origin_question = data["question"]
        # marker_prev_answer = "You are given the full history of your previous attempts and feedback provided."
        if round >= 1: # if previously exists then we append full question
            # small fix: remove the "Question" string at front
            question_part = origin_question # + "\nQuestion: " + origin_question.split(marker_prev_answer)[0]
            previous = question_part
        else: # if this is the first attempt
            question_part = origin_question
            previous = "Question: " + question_part + "\nChoices:\n"
            for i in range(len(data['options'])):
                previous += f"({chr(ord('A') + i)}) " + data['options'][i] + "\n"
            previous += "Answer:"
        return previous
    
    elif dataset_name == "custom_simple":
        # problematic for future iterations
        origin_question = data["question"]
        # this has been changed
        return origin_question
    
    elif dataset_name == "hex":
        previous = data['question']
        return previous

def get_demonstrations(dataset_name, category):
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
    elif dataset_name == "gsm8k_symbolic":
        messages_gsm8k_symbolic = []
        rand_list_from_train = np.random.choice(gsm8k_symbolic_datalist, 9, replace=False)
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
            messages_gsm8k_symbolic.extend(l)
        return messages_gsm8k_symbolic
    elif dataset_name == "math":
        messages_math = [{"role": "system", "content": "You are a smart assistant that solves math problems. Please think step by step to solve the problem. If you think you're ready to output the answer, you can wrap your answer with \\boxed{}. Please follow this format"}]
        # messages_math.append({"role": "user", "content": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$."})
        # messages_math.append({"role": "assistant", "content": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct."})
        # messages_math.append({"role": "user", "content": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$"})
        # messages_math.append({"role": "assistant", "content": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.\nFinal Answer: The final answer is $24$. I hope it is correct."})
        # messages_math.append({"role": "user", "content": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?"})
        # messages_math.append({"role": "assistant", "content": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct."})
        # messages_math.append({"role": "user", "content": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero."})
        # messages_math.append({"role": "assistant", "content": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."})
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
    
    elif dataset_name == "aime_2024":
        messages_aime = [{"role": "system", "content": "You are a smart assistant that solves math problems. Please think step by step to solve the problem. If you think you're ready to output the answer, you can wrap your answer with \\boxed{}. Please follow this format"}]
        return messages_aime
    
    elif dataset_name == "trivia_qa":
        messages_triviaqa = [{"role": "system", "content": "You are a smart assistant that solves trivia questions. If you think you're ready to output the answer, you can just output an answer."}]
        rand_list_from_train = np.random.choice(triviaqa_datalist, 5, replace=False)
        for data in rand_list_from_train:
            l = []
            d = {"role": "user", "content": data['question']}
            l.append(d)
            l.append({"role": "assistant", "content": data['answer']["normalized_aliases"][0]})
            messages_triviaqa.extend(l)
        return messages_triviaqa # check
    
    elif dataset_name == "pop_qa":
        messages_popqa = [{"role": "system", "content": "You are a smart assistant that answers fact-based questions. If you think you're ready to output the answer, you can just output an answer."}]
        
        rand_list_from_popqa = np.random.choice(popqa_datalist, 5, replace=False)

        for data in rand_list_from_popqa:
            l = []
            l.append({"role": "user", "content": data['question']})
            l.append({"role": "assistant", "content": data['possible_answers'][0]})  # assumes at least one answer
            messages_popqa.extend(l)
        
        return messages_popqa

    elif dataset_name == "mmlu": 
        messages_mmlu = [{
            "role": "system",
            "content": (
                f"The following are multiple-choice questions about {category}. "
                "Let's think step by step. Please explain your reasoning clearly as you work toward the answer. "
                "When you're ready, conclude your answer with the phrase: \"The answer is (X)\", "
                "where X is the correct letter choice. Make sure to always include parentheses around the letter."
            )        
        }] # same way for extract the answer
        rand_list_from_train = np.random.choice(mmlu_datalist_all[category], 5, replace=False)
        i = 0
        for data in rand_list_from_train: # since no cot content exists, we just use answer
            l = []
            d = {
                "role": "user",
                "content": data['question'] + "\n\nChoices: " + '\n'.join(data["choices"])
            }
            for j in range(len(rand_list_from_train[i]['choices'])):
                d['content'] += f"({chr(ord('A') + j)}) " + rand_list_from_train[i]['choices'][j] + "\n"
            d['content'] = d['content'] + 'Answer: '
            l.append(d)
            num = data['answer']
            idx_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            ans_num = idx_letter[num]
            l.append({"role": "assistant", "content": "The answer is " + "(" + ans_num + ")"})
            messages_mmlu.extend(l)
            i += 1
        # print(messages_mmlu)
        return messages_mmlu
    elif dataset_name == "mmlu_pro":
        messages_mmlu_pro = [{
            "role": "system",
            "content": (
                f"The following are multiple-choice questions about {category}. "
                "Let's think step by step. Please explain your reasoning clearly as you work toward the answer. "
                "When you're ready, conclude your answer with the phrase: \"The answer is (X)\", "
                "where X is the correct letter choice. Make sure to always include parentheses around the letter."
            )
        }] # same way for extract the answer
        data_list = np.random.choice(mmlu_datalist[category], 5, replace=False)
        for i in range(5):
            d = {"role": "user", "content": "Question: " + data_list[i]['question'] + "\nChoices: "}
            for j in range(len(data_list[i]['options'])):
                d['content'] +=  f"({chr(ord('A') + j)}) " + data_list[i]['options'][j] + "\n"
            d['content'] = d['content'] + 'Answer: '
            # for i in range(len(data['options'])):
            #    previous += f"({chr(ord('A') + i)}) " + data['options'][i] + "\n"
            # previous += "Answer:
            messages_mmlu_pro.append(d)
            cot_reasoning = data_list[i]["cot_content"].replace("A: ", "")
            # cot_reasoning_splits = cot_reasoning.split(". ")
            # for cot_reasoning_split in cot_reasoning_splits:
            #     messages_mmlu_pro.append({"role": "assistant", "content": cot_reasoning_split})
            messages_mmlu_pro.append({"role": "assistant", "content": cot_reasoning})
        return messages_mmlu_pro
    elif dataset_name == "gpqa":# revise 0 shots
        '''
        messages_gpqa = [{
            "role": "system",
            "content": (
                "The following are multiple-choice questions. Think step by step. "
                "When you're done, return your response as a JSON object with two fields:\n\n"
                "- \"explanation\": a string explaining your reasoning\n"
                "- \"answer\": the final choice as one of \"A\", \"B\", \"C\", or \"D\"\n\n"
                "Example format:\n"
                "{\n  \"explanation\": \"...your step-by-step reasoning...\",\n  \"answer\": \"C\" \n}"
            )
        }] 
        '''
        messages_gpqa = [{
            "role": "system",
            "content": "The following are multiple-choice questions. Let's think step by step. Please explain your reasoning clearly as you work toward the answer. When you're ready, please finish your answer with \"The answer is (X)\" where X is the correct letter choice. Make sure to always include parentheses around the letter. "
        }]
        
        '''
        messages_gpqa = [{
            "role": "system",
            "content": "The following are multiple-choice questions. Let's think step by step. Please first output the answer with \"The answer is (X)\" where X is the correct letter choice. Make sure to always include parentheses around the letter. Then, Please explain your reasoning clearly as you work toward the answer."
        }]
        '''
        return messages_gpqa
    elif dataset_name == "custom_simple": # zero shots
        message_custom = [{# need to revise back
            "role":"system",
            "content":"You are a smart assistant in solving multiplication questions. Please think step by step. If you think you're ready to output the answer, you can wrap your answer with \\boxed{}. Please follow this format. "
        }]
        '''
        fewshots_path = dataset_files["fewshots"]
        with open(fewshots_path, 'r') as f:
            fewshots = json.load(f)  # load as a list of messages
        message_custom.extend(fewshots)
        '''
        return message_custom

    elif dataset_name == "hex": 
        message_hex = [{
            "role":"system",
            "content":"You are a smart assistant in solving hexadecimal multiplication questions. Please think step by step. If you think you're ready to output the answer, you can wrap your answer with \\boxed{}. Please follow this format. "
        }]
        return message_hex

def get_normalized_answer(dataset_name, data):
    if dataset_name == "arc" or dataset_name == "ecqa":
        return data['answerKey']
    elif dataset_name == "gsm8k" or dataset_name == "gsm8k_symbolic":
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
    elif dataset_name == "pop_qa":
        print(data['possible_answers'][0])
        return data['possible_answers'][0]
    elif dataset_name == "aime_2024":
        return str(data['Answer'])
    elif dataset_name == "mmlu":
        number = data['answer']
        index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        ans = index_to_letter[number]
        return ans # get a letter output
    elif dataset_name == "mmlu_pro":
        return data['answer']
    # TODO: double check function is as intended
    elif dataset_name == "gpqa": #revised
        number = data['answer']
        index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        ans = index_to_letter[number]
        return ans # get a letter output
    elif dataset_name == "custom_simple":
        solution = data['answer']
        return str(solution)
    elif dataset_name == "hex":
        return data['answer']
    
def get_raw_answer(dataset_name, data):
    if dataset_name == "gpqa": #revised
        number = data['answer']
        ans = data["options"][number]
    return ans # get a raw output


def get_dataset_key(dataset_name):
    if dataset_name == "arc" or dataset_name == "ecqa" or dataset_name == "gsm8k" or dataset_name == "mmlu_pro" or dataset_name == "gsm8k_symbolic" or dataset_name == "mmlu":
        return "question"
    elif dataset_name == "math":
        return "problem"
    elif dataset_name == "aime_2024":
        return "Problem"
    elif dataset_name == "proofwriter":
        return "context"
    elif dataset_name == "trivia_qa" or dataset_name == "pop_qa":
        return "question"
    elif dataset_name == "gpqa":
        return "question" # TODO: check the revision from Question to question
    elif dataset_name == "custom_simple" or "hex":
        return "question"
    

def get_process_answer(dataset_name, data):
    if dataset_name == "gsm8k" or dataset_name == "gsm8k_symbolic":
        return data['answer']
    elif dataset_name == "math":
        return data["solution"]
    elif dataset_name == "trivia_qa":
        return data['answer']["normalized_value"]
    elif dataset_name == "pop_qa":
        return data['possible_answers'][0]
    elif dataset_name == "aime_2024":
        return data['Solution']
    elif dataset_name == "mmlu":
        number = data['answer'] # map numbers to letters
        index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        ans = index_to_letter[number]
        return ans # get a letter output
    elif dataset_name == "mmlu_pro":
        return data['answer']
    elif dataset_name == "gpqa": 
        # TODO: to run GPQA answer + solution correctly, we need to add the --use_process_feedback
        # the results we previously got is also correct since I manually add the explanantion before in the get_normalized_answer part
        # but then I discovered this function and the revise to use this instead.
        number = data['answer'] # map numbers to letters
        index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        ans = index_to_letter[number]
        final_ans = "The answer is: " + ans + ". The explanation is: " + data['Explanation'] # use this if we use feedback with solution
        return final_ans
    elif dataset_name == "custom_simple":
        # return str(data['answer'])
        return data["Explanation"]
    elif dataset_name == "hex":
        return data["Explanation"]

def get_normalized_prediction(dataset_name, prediction):
    if dataset_name == "arc" or dataset_name == "ecqa" or dataset_name == "proofwriter":
        normalized_prediction = prediction.strip().replace(": ", "")
        return normalized_prediction
    elif dataset_name == "gsm8k" or dataset_name == "gsm8k_symbolic":
        return extract_and_convert_number_real(prediction)
    elif dataset_name == "math":
        res = remove_boxed(last_boxed_only_string(prediction))
        try:
            res = strip_string(res)
        except:
            pass
        return res
    elif dataset_name == "aime_2024":
        res = remove_boxed(last_boxed_only_string(prediction))
        try:
            res = strip_string(res)
        except:
            pass
        return res
    elif dataset_name == "trivia_qa":
        return normalize_answer(prediction)
    
    elif dataset_name == "pop_qa":
        return prediction
    
    # TODO: please double check the implementation. I've created test cases for them in test_re.py
    if dataset_name == "mmlu":  # extract the formatted answer
        pattern = re.compile(
            r"(?i)"                              # Case-insensitive
            r"(?:answer\s*(?:is\s*:|is|:)?)\s*"  # Match 'answer', 'answer is', 'answer:', or 'answer is:'
            r"(?:"                              
            r"\(\(\s*([A-D])\s*\)\)"            # Match ((A))
            r"|"
            r"\(\s*([A-D])\s*\)"                # Match (A)
            # r"|"
            # r"\b([A-D])\b"                      # Match A-D as whole word only
            r")"
        )
        prediction = prediction.replace("*", "")
        match = pattern.search(prediction)
        if match:
            answer = match.group(1) or match.group(2)
            return answer.upper()
        return "X"  # not found
    
    elif dataset_name == "mmlu_pro":  # extract the formatted answer
        pattern = re.compile(
            r"(?i)"                              # Case-insensitive
            r"(?:answer\s*(?:is\s*:|is|:)?)\s*"  # Match 'answer', 'answer is', 'answer:', or 'answer is:'
            r"(?:"                              
            r"\(\(\s*([A-J])\s*\)\)"            # Match ((A))
            r"|"
            r"\(\s*([A-J])\s*\)"                # Match (A)
            # r"|"
            # r"\b([A-J])\b"                      # Match A-J as whole word only
            r")"
        )
        prediction = prediction.replace("*", "")
        match = pattern.search(prediction)
        if match:
            answer = match.group(1) or match.group(2)
            return answer.upper()
        return "X"  # not found
        
    elif dataset_name == "gpqa":
     
        pattern = re.compile(
            r"(?i)"                              # Case-insensitive
            r"(?:answer\s*(?:is\s*:|is|:)?)\s*"  # Match 'answer', 'answer is', 'answer:', or 'answer is:'
            r"(?:"                              
            r"\(\(\s*([A-D])\s*\)\)"            # Match ((A))
            r"|"
            r"\(\s*([A-D])\s*\)"                # Match (A)
            # r"|"
            # r"\b([A-D])\b"                      # Match A-D as whole word only
            r")"
        )
        prediction = prediction.replace("*", "")
        match = pattern.search(prediction)
        if match:
            answer = match.group(1) or match.group(2)
            return answer.upper()
        return "X"  # not found


    elif dataset_name == "custom_simple":
        res = last_boxed_only_string(prediction)
        if res:
            res = remove_boxed(res)
            try:
                res = strip_string_mult(res)
            except Exception as e:
                print(f"Error processing result in multiplication extaction: {e}")
        return res if res else ""

    elif dataset_name == "hex":
        res = last_boxed_only_string(prediction)
        if res:
            res = remove_boxed(res)
            try:
                res = strip_string_mult(res)
            except Exception as e:
                print(f"Error processing result in multiplication extaction: {e}")
        return res if res else ""


def flatten_list(new_messages):
    messages = []
    for message in new_messages:
        messages.append(message["role"] + ": " + message["content"])
    return "\n".join(messages)


async def is_equivalent(dataset, item, data):
    if dataset == "math":
        try:
            # I think here is problematic
            a, b = normalize_final_answer(item["normalized_prediction"][0]), normalize_final_answer(get_normalized_answer(dataset, data))
            '''
            print(a,b)
            print(a == b)
            print("repr:", repr(a), repr(b))
            print("lengths:", len(a), len(b))
            for i, (ca, cb) in enumerate(zip(a, b)):
                if ca != cb:
                    print(f"  diff at index {i!r}: {ca!r} vs {cb!r} (ord {ord(ca)} vs {ord(cb)})")
            print("the length of the item is: ", len(item["normalized_prediction"]))
            '''
            if len(item["normalized_prediction"]) >= 1 and (a.strip() == b.strip()): # removed is_equiv(a, b) or 
                return True
            if len(item["normalized_prediction"]) >= 1 and item["normalized_prediction"][0] == item["normalized_answer"]:
                return True
            if len(item["normalized_prediction"]) >= 1 and a == b:
                return True
            if len(item["normalized_prediction"]) >= 1 and is_equiv(a, b): # fixed this function
                return True
        except:
            return False
        return False
    if dataset == "aime_2024":
        try:
            # I think here is problematic
            a, b = normalize_final_answer(item["normalized_prediction"][0]), get_normalized_answer(dataset, data)
            '''
            print(a,b)
            print(a == b)
            print("repr:", repr(a), repr(b))
            print("lengths:", len(a), len(b))
            for i, (ca, cb) in enumerate(zip(a, b)):
                if ca != cb:
                    print(f"  diff at index {i!r}: {ca!r} vs {cb!r} (ord {ord(ca)} vs {ord(cb)})")
            print("the length of the item is: ", len(item["normalized_prediction"]))
            '''
            if len(item["normalized_prediction"]) >= 1 and (a.strip() == b.strip()): # removed is_equiv(a, b) or 
                return True
            if len(item["normalized_prediction"]) >= 1 and item["normalized_prediction"][0] == item["normalized_answer"]:
                return True
            if len(item["normalized_prediction"]) >= 1 and a == b:
                return True
            if len(item["normalized_prediction"]) >= 1 and is_equiv(a, b): # fixed this function
                return True
        except:
            return False
        return False
    elif dataset == "arc" or dataset == "gsm8k" or dataset == "gpqa" or dataset == "mmlu_pro" or dataset == "gsm8k_symbolic" or dataset == "mmlu" or dataset == "custom_simple" or dataset == "hex":
        if len(item["normalized_prediction"]) >= 1 and item["normalized_prediction"][0] == item["normalized_answer"]:
            return True
        else:
            return False
    elif dataset == "trivia_qa":
        if len(item["normalized_prediction"]) >= 1 and item["normalized_prediction"][0] in data['answer']['normalized_aliases']:
            return True
        else:
            return False

    elif dataset == "pop_qa":
        # no prediction → incorrect
        if not item.get("normalized_prediction"):
            return False

        question   = data.get("original_question", "").strip()
        prediction = item["normalized_prediction"][0].strip()
        possible_norm = [normalize_answer(ans) for ans in data["possible_answers"]]
        verdict, raw_verdict = await judge_pop_qa(question, prediction, possible_norm)
        # cache the verdict also
        print("using judge model comment: ", raw_verdict)
        if "judge_responses" not in data:
            data["judge_responses"] = []
        data["judge_responses"].append({
            "raw_verdict": raw_verdict
        })
        return verdict
    

def get_normalized_predictions(dataset, response_list):
    normalized_prediction_list = []
    if dataset == "gsm8k" or dataset == "gsm8k_symbolic":
        for term in gsm8k_list_of_end_prompts:
            try:
                for response in response_list:
                    if term in response:
                        prediction = response.split(term)[1].replace('\n', ' ').strip()
                        normalized_prediction = get_normalized_prediction(dataset, prediction)
                        if normalized_prediction == "":
                            pass
                        else:
                            normalized_prediction_list.append(normalized_prediction)
                            break
                if len(normalized_prediction_list) != 0:
                    break
            except Exception as e:
                print(e)
                print("Error")
    elif dataset == "math" or dataset == "arc" or dataset == "trivia_qa" or dataset == "custom_simple" or dataset == "aime_2024" or dataset == "pop_qa" or dataset == "hex":
        for response in response_list:
            try:
                prediction = response
                normalized_prediction = get_normalized_prediction(dataset, prediction)
                normalized_prediction_list.append(normalized_prediction)
            except Exception as e:
                print(e)
                print("Error")
    elif dataset == "mmlu_pro":
        for response in response_list:
            try:
                prediction = response
                normalized_prediction = get_normalized_prediction(dataset, prediction)
                normalized_prediction_list.append(normalized_prediction)
            except:
                print("Error")
    elif dataset == "mmlu" or dataset == "gpqa":
        for response in response_list:
            try:
                prediction = response
                normalized_prediction = get_normalized_prediction(dataset, prediction)
                normalized_prediction_list.append(normalized_prediction)
            except:
                print("Error")
    return normalized_prediction_list


async def call_vllm_server(agent_model, new_messages, temperature, n, tokenizer, base_url, ports, cache, type=None, dataset=None, round=None, logprobs=None):
    if n != 1:
        raise ValueError("n must be 1")
    # res = list(cache.search(
    #     prompt=flatten_list(new_messages),
    #     model=agent_model,
    #     temperature=temperature
    # ))
    # if len(res) > 0 and len(res) == 1:
    #     print("Cache hit")
    #     return res[0]['response']
    # elif len(res) > 0 and len(res) > 1:
    #     raise ValueError("Multiple results found")
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
        if logprobs is not None:
            prob = sum(item["logprob"] for item in agent_response['choices'][0]['logprobs']['content'][:-1]) / (len(agent_response['choices'][0]['logprobs']['content']) - 1)
        else:
            prob = None
        agent_response = agent_response['choices'][0]['message']['content']
    except:
        agent_response = ""
        prob = None
    # Store responses
    if type is None or dataset is None or round is None:
        raise ValueError("Type or dataset is None")
    # cache.store(
    #     prompt=flatten_list(new_messages),
    #     response=agent_response,
    #     model=agent_model,
    #     temperature=temperature,
    #     type=type,
    #     dataset=dataset,
    #     round=round
    # )
    
    return (agent_response, prob)

# default version of calling the reasoner model
async def call_vllm_server_reasoner(agent_model, new_messages, temperature, n, tokenizer, base_url, ports, cache, type=None, dataset=None, round=None, logprobs=None):
    if n != 1:
        raise ValueError("n must be 1")

    url = get_url(base_url, ports)
    content = {
        "model": agent_model,
        "messages": new_messages,
        "max_tokens": 32768,
        "temperature": 0.6,       
        "extra_body": {
            "top_k": 20,
            "top_p": 0.95,
            "min_p": 0
        },
        "stop_token_ids": [128001, 128009, tokenizer.eos_token_id],
        "best_of": n,
        "n": n,
        "logprobs": logprobs,
        "seed": 14,
        "chat_template": tokenizer.chat_template
    }
    headers = {
        "Content-Type": "application/json"
    }
    session_timeout = aiohttp.ClientTimeout(total=1200000,sock_connect=120000,sock_read=120000)

    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        async with session.post(url, headers=headers, json=content) as agent_response:
            try:
                agent_response.raise_for_status()
                agent_response = await agent_response.json()
            except Exception as e:
                print(e)
                print("Error in calling remote agent server")
    
    try:
        if logprobs is not None:
            prob = sum(item["logprob"] for item in agent_response['choices'][0]['logprobs']['content'][:-1]) / (len(agent_response['choices'][0]['logprobs']['content']) - 1)
        else:
            prob = None
        agent_response = agent_response['choices'][0]['message']['content']
        reasoning_content = agent_response.choices[0].message.reasoning_content
    except:
        agent_response = ""
        reasoning_content = ""
        prob = None
    # Store responses
    if type is None or dataset is None or round is None:
        raise ValueError("Type or dataset is None")
    
    return (agent_response, prob, reasoning_content)

# we rely on default temp at here
async def call_vllm_server_reasoner_json(
    agent_model, new_messages, temperature, n, tokenizer,
    base_url, ports, cache, type=None, dataset=None, round=None, logprobs=None
):
    if n != 1:
        raise ValueError("n must be 1")

    url = get_url(base_url, ports)
    content = {
        "model": agent_model,
        "messages": new_messages,
        "max_tokens": 32768,
        "stop_token_ids": [128001, 128009, tokenizer.eos_token_id],
        "best_of": n,
        "n": n,
        "temperature": 0.6,
        "extra_body": {
            "top_k": 20,
            "top_p": 0.95,
            "min_p": 0
        },
        "response_format": {
            "type": "json_object",
            "properties": {
                "explanation": {"type": "string"},
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]}
            },
            "required": ["answer"]
        },
        "logprobs": logprobs,
        "seed": 14,
        "chat_template": tokenizer.chat_template
    }

    headers = {
        "Content-Type": "application/json"
    }
    session_timeout = aiohttp.ClientTimeout(total=60000, sock_connect=6000, sock_read=6000)

    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        async with session.post(url, headers=headers, json=content) as agent_response:
            try:
                agent_response.raise_for_status()
                agent_response = await agent_response.json()
            except Exception as e:
                print(e)
                print("Error in calling remote agent server")
                agent_response = None

    try:
        if logprobs is not None and agent_response:
            prob = sum(
                item["logprob"] for item in agent_response['choices'][0]['logprobs']['content'][:-1]
            ) / (len(agent_response['choices'][0]['logprobs']['content']) - 1)
        else:
            prob = None

        content_raw = agent_response['choices'][0]['message']['content']

        # Attempt to parse structured response
        try:
            parsed = json.loads(content_raw)
            explanation = parsed.get("explanation", "").strip()
            answer = parsed.get("answer", "").strip().upper()

            # Combine explanation and answer in legacy-friendly format
            agent_response = f"{explanation}\n\nThe answer is ({answer})"
        except Exception as e:
            print(f"Warning: JSON parsing failed. Using raw output. Error: {e}")
            agent_response = content_raw

    except Exception as e:
        print(f"Top-level extraction failed: {e}")
        agent_response = ""
        prob = None

    if type is None or dataset is None or round is None:
        raise ValueError("Type or dataset is None")

    return (agent_response, prob)



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

def mask_answer_in_string(input_string, ground_truth):
    ground_truth_str = str(ground_truth)  # Ensure it's a string
    safe_ground_truth = re.escape(ground_truth_str)  # Escape special regex characters
    masked_string = re.sub(rf'{safe_ground_truth}', '[masked]', input_string)
    return masked_string

def mask_answers_in_trivia_qa(feedback_string, data):
    """
    Mask all answers (aliases and normalized_aliases) in the feedback string.
    Uses word boundaries to ensure only complete words/phrases are masked.
    """
    all_answers = data['answer']['aliases'] + data['answer']['normalized_aliases']
    all_answers = list(set(filter(None, map(str.strip, all_answers))))  # Remove empty strings
    all_answers.sort(key=len, reverse=True)  # Process longer phrases first
    
    masked_feedback = feedback_string
    for answer in all_answers:
        safe_answer = re.escape(answer)
        pattern = rf'\b{safe_answer}\b'
        masked_feedback = re.sub(pattern, '[masked]', masked_feedback, flags=re.IGNORECASE)
    
    return masked_feedback

def mask_answers_in_pop_qa(feedback_string, data):
    """
    Mask all answers (aliases and normalized_aliases) in the feedback string.
    Uses word boundaries to ensure only complete words/phrases are masked.
    """
    all_answers = data['possible_answers']
    norm_data = [normalize_answer(d) for d in all_answers]
    all_answers = all_answers + norm_data
    all_answers = list(set(filter(None, map(str.strip, all_answers))))  # Remove empty strings
    all_answers.sort(key=len, reverse=True)  # Process longer phrases first
    
    masked_feedback = feedback_string
    for answer in all_answers:
        safe_answer = re.escape(answer)
        pattern = rf'\b{safe_answer}\b'
        masked_feedback = re.sub(pattern, '[masked]', masked_feedback, flags=re.IGNORECASE)
    
    return masked_feedback

def mask_answer_in_string_math(input_string, ground_truth):
    ground_truth_str = str(ground_truth)
    safe_ground_truth = re.escape(ground_truth_str)

    # First: mask the entire \boxed{ground_truth}
    masked_string = re.sub(rf'\\boxed\s*\{{\s*{safe_ground_truth}\s*\}}', '[masked]', input_string)

    # Then: mask standalone ground_truth
    masked_string = re.sub(rf'\b{safe_ground_truth}\b', '[masked]', masked_string)

    return masked_string

def mask_answer_in_string_hex(input_string: str, ground_truth: str) -> str:
    gt = ground_truth.upper()
    safe_gt = re.escape(gt)

    # 1) Mask LaTeX boxed forms:
    masked = re.sub(
        rf'\\boxed\s*\{{\s*{safe_gt}\s*\}}',
        '[masked]',
        input_string,
        flags=re.IGNORECASE
    )

    # 2) Mask standalone occurrences, ignoring case and punctuation boundaries:
    masked = re.sub(
        rf'(?<![A-Za-z0-9]){safe_gt}(?![A-Za-z0-9])',
        '[masked]',
        masked,
        flags=re.IGNORECASE
    )

    return masked

def mask_answer_in_string_mcq_case_sensitive(input_string, ground_truth_letter):
    # Only allow uppercase A–E or similar; reject lowercase inputs
    if not ground_truth_letter.isupper():
        return input_string  # no-op if not uppercase

    safe_letter = re.escape(ground_truth_letter)

    # Case-sensitive patterns (no re.IGNORECASE)
    patterns = [
        rf'\(\s*{safe_letter}\s*\)',              # (C)
        rf'\\boxed\s*\{{\s*{safe_letter}\s*\}}',  # \boxed{C}
        rf'\"{safe_letter}\"',                    # "C"
        rf'\*\*{safe_letter}\*\*',                # **C**
        rf'\b{safe_letter}\b'                     # plain C
    ]

    masked_string = input_string
    for pattern in patterns:
        masked_string = re.sub(pattern, '[masked]', masked_string)

    return masked_string

def mask_answer_in_string_strict(input_string, ground_truth, dataset_name, data):
    # ground_truth_str = str(ground_truth)  # Ensure it's a string
    # safe_ground_truth = re.escape(ground_truth_str)  # Escape special regex characters
    # masked_string = re.sub(rf'{safe_ground_truth}', 'XXX', input_string, flags=re.IGNORECASE)
    
    # Also mask explicit answer statements with different formats
    masked_string = re.sub(r'(?i)(The answer is|Correct answer:|Answer:|answer is:|answer is |answer: )\s+([A-D0-9]+)', 'The answer is XXX', input_string, flags=re.IGNORECASE)
    
    # Get raw answer from dataset and mask it
    raw_answer = get_raw_answer(dataset_name, data)
    if raw_answer:
        safe_raw_answer = re.escape(str(raw_answer))  # Escape for regex
        masked_string = re.sub(rf'{safe_raw_answer}', 'XXX', masked_string, flags=re.IGNORECASE)
    
    return masked_string

'''
# TODO: newly added for multiplication questions
def mask_answer_in_string_arith(input_string, ground_truth):
    ground_truth_str = str(ground_truth)

    # this will mask those with 1,000,000 "," between numbers
    comma_formatted = re.sub(r"(\d)(?=(\d{3})+$)", r"\1,", ground_truth_str)

    # ensure we only match the exact number with or without commas
    pattern = rf'\b{re.escape(ground_truth_str)}\b|\b{re.escape(comma_formatted)}\b'

    # replace occurrences with "<answer masked>"
    masked_string = re.sub(pattern, '<answer masked>', input_string)

    return masked_string
'''

def mask_answer_in_string_arith(input_string, ground_truth, intermediate_steps=None):
    """
    Mask each value (ground truth and intermediate steps) with unique placeholders like [Mask1], [Mask2], etc.
    """
    ground_truth_str = str(ground_truth)
    intermediate_steps = intermediate_steps or []

    # Create ordered list of values to mask
    all_values = intermediate_steps + [ground_truth]
    value_to_mask = {}

    # Assign unique mask tokens: [Mask1], [Mask2], ...
    for i, val in enumerate(all_values, 1):
        clean_val = str(val).replace(",", "")
        value_to_mask[clean_val] = f"[Mask{i}]"

    # Define replacer function
    def replacer(match):
        candidate = match.group(0)
        candidate_cleaned = candidate.replace(",", "")
        return value_to_mask.get(candidate_cleaned, candidate)

    # Apply masking
    masked_string = re.sub(r'\b[\d,]+\b', replacer, input_string)
    return masked_string


def mask_answer_in_string_mcq(input_string, ground_truth): # possible to use if we cannot eliminate leak of answer choice
    ground_truth_str = re.escape(str(ground_truth))
    pattern = rf'\(\s*{ground_truth_str}\s*\)'
    masked_string = re.sub(pattern, '<answer masked>', input_string)
    return masked_string

def check_if_ground_truth_exists(input_string, ground_truth):
    # return True if ground truth exists
    ground_truth_str = str(ground_truth)
    match = re.search(rf'\b{re.escape(ground_truth_str)}\b', input_string)
    # match = re.search(rf'\b{ground_truth_str}\b', input_string)
    return match is not None

# not useful for now
def check_if_ground_truth_exists_mcq(input_string, ground_truth):

    ground_truth_str = str(ground_truth)
    # debug stuff, not useful now
    if input_string is None:
        print("input_string is None")
        return False
    elif not isinstance(input_string, str):
        print(f"input_string is not of type str: {type(input_string)}")
        return False

    match = re.search(rf'\({ground_truth_str}\)', input_string)
    return match is not None


def extract_predictions(dataset, response_list):
    normalized_prediction_list = []
    if dataset == "gsm8k":
        for term in gsm8k_list_of_end_prompts:
            try:
                for response in response_list:
                    if term in response:
                        prediction = response.split(term)[1].replace('\n', ' ').strip()
                        normalized_prediction = get_normalized_prediction(dataset, prediction)
                        if normalized_prediction == "":
                            pass
                        else:
                            normalized_prediction_list.append(normalized_prediction)
                            break
                if len(normalized_prediction_list) != 0:
                    break
            except:
                print("Error")
    elif dataset == "math" or dataset == "custom_simple" or dataset == "aime_2024":
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
    elif dataset == "trivia_qa" or dataset == "pop_qa":
        for response in response_list:
            try:
                prediction = response
                normalized_prediction = get_normalized_prediction(dataset, prediction)
                normalized_prediction_list.append(normalized_prediction)
            except:
                print("Error")
    elif dataset == "mmlu" or dataset == "gpqa": # try revised
        for response in response_list:
            try:
                prediction = response
                normalized_prediction = get_normalized_prediction(dataset, prediction)
                normalized_prediction_list.append(normalized_prediction)
            except:
                print("Error")
    elif dataset == "mmlu_pro": # try
        for response in response_list:
            try:
                prediction = response
                normalized_prediction = get_normalized_prediction(dataset, prediction)
                normalized_prediction_list.append(normalized_prediction)
            except:
                print("Error")
    return normalized_prediction_list

# TODO: please double check
def generate_question(dataset, data, round=0): # please check do we need to add the original question once again at the end for completness? see example below
    # get previous is used for prompting the model while generate_question is used for record the question
    if dataset == "arc":
        question = data[get_dataset_key(dataset)] + "\n\nChoices: " + '\n'.join(data["choices"]["text"])
    elif dataset == "math" or dataset == "trivia_qa" or dataset == "pop_qa" or dataset == "gsm8k" or dataset == "custom_simple" or dataset == "custom_hard" or dataset == "aime_2024": # or dataset == "gpqa"
        question = data[get_dataset_key(dataset)]
    elif dataset == "mmlu": # make the style consistent
        if round == 0: # only add the choices at last in the first round
            question = data[get_dataset_key(dataset)] + "\nChoices: " + "\n".join([f"({chr(ord('A') + i)}) {option}" for i, option in enumerate(data['choices'])])
        else:
            question = data[get_dataset_key(dataset)]
    elif dataset == "mmlu_pro":
        if round == 0:
            question = data[get_dataset_key(dataset)] + "\nChoices: " + "\n".join([f"({chr(ord('A') + i)}) {option}" for i, option in enumerate(data['options'])])
        else:
            question = data[get_dataset_key(dataset)]
    elif dataset == "gpqa":
        if round == 0:
            question = data[get_dataset_key(dataset)] + "\nChoices: " + "\n".join([f"({chr(ord('A') + i)}) {option}" for i, option in enumerate(data['options'])])
        else:
            question = data[get_dataset_key(dataset)]
        # example: question = data[get_dataset_key(dataset)]  + original question + "\nChoices: " + '\n'.join(data['options'])
    return question



