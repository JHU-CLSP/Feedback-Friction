import re
import string
import random
import datasets
import asyncio
import aiohttp
import numpy as np
import json
from datetime import datetime
from dataset_specific_utils import normalize_final_answer, remove_boxed, last_boxed_only_string, is_equiv, extract_and_convert_number_real, strip_string, normalize_answer, gsm8k_list_of_end_prompts


def get_url(base_url, ports):
    return random.choice(base_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'


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
    elif dataset_name == "mmlu": # adding mmlu dataset
        ds = datasets.load_dataset("cais/mmlu", "all")
        data_list = list(ds['test'])
        global mmlu_datalist_all
        """
        mmlu_datalist_all = list(ds['auxiliary_train'])
        if mode == "test":
            return data_list
        elif mode == "train":
            return mmlu_datalist_all
        """
        mmlu_datalist_all = {}
        mmlu_datalist_all_ls = list(ds['validation'])
        for data in mmlu_datalist_all_ls:
            if data['subject'] not in mmlu_datalist_all:
                mmlu_datalist_all[data['subject']] = []
            mmlu_datalist_all[data['subject']].append(data)
        if mode == "test":
            return data_list # all test data
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
            return data_list # all test data
        elif mode == "train":
            return mmlu_datalist # a dictionary that contains 12 subcategories with 5 questoins in each 
        
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
        custom_list = load_dataset("custom_simple")
        return custom_list
        
# load the costumsized dataset
def load_dataset(dataset_name):
    dataset_files = {
        "custom_simple": "/scratch/dkhasha1/bzhang90/Self-InPerfect/digits_buckets/multiplication_questions_7d_part1.jsonl",
    }
    
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

# TODO: do we need to reprompt the model of the original question at the end again? see example below:
def get_previous(dataset_name, data):
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
    elif dataset_name == "trivia_qa":
        return data['question']
    elif dataset_name == "mmlu":
        previous = "Question: " + data['question'] + "\nChoices: "
        for i in range(len(data['choices'])):
            previous += f"({chr(ord('A') + i)}) " + data['choices'][i] + "\n"
        previous += "Answer:"
        # previous = "Question: " + data['question']  + "\nChoices: " + '\n'.join(data['choices']) + '\nAnswer:'
        return previous
    elif dataset_name == "mmlu_pro":
        previous = "Question: " + data['question'] + "\nChoices: "
        # TODO:
        # example: previous = "Question: " + data['question'] + original question + "\nChoices: ""
        for i in range(len(data['options'])):
            previous += f"({chr(ord('A') + i)}) " + data['options'][i] + "\n"
        previous += "Answer: "
        return previous
    elif dataset_name == "gpqa": #revised
        
        previous = "Question: " + data['question'] + "\nChoices:\n"
        for i in range(len(data['options'])):
            previous += f"({chr(ord('A') + i)}) " + data['options'][i] + "\n"
        previous += "Answer:"
        
        return previous
    elif dataset_name == "custom_simple":
        previous = "Question: " + data['question'] + '\nAnswer: '
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
    elif dataset_name == "mmlu": 
        messages_mmlu = [{
            "role": "system",
            "content": "The following are multiple-choice questions (with answers) about " + category + ". Let's think step by step. You will only generate one sentence that extends the reasoning trajectory that solves the question given the question and partial answer reasoning trajectory. When you're ready, please finish your answer with \"The answer is (X)\" where X is the correct letter choice. Please always include the parentheses around the letter choice."
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
                d['content'] += f"({chr(ord('A') + i)}) " + rand_list_from_train[i]['choices'][j] + "\n"
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
            "content": "The following are multiple-choice questions (with answers) about " + category + ". Let's think step by step. You will only generate one sentence that extends the reasoning trajectory that solves the question given the question and partial answer reasoning trajectory. When you're ready, please finish your answer with \"The answer is (X)\" where X is the correct letter choice. Please always include the parentheses around the letter choice."
        }]
        data_list = np.random.choice(mmlu_datalist[category], 5, replace=False)
        for i in range(5):
            d = {"role": "user", "content": "Question: " + data_list[i]['question'] + "\nChoices: "}
            for j in range(len(data_list[i]['options'])):
                d['content'] +=  f"({chr(ord('A') + i)}) " + data_list[i]['options'][j] + "\n"
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
        messages_gpqa = [{
            "role": "system",
            "content": "The following are multiple-choice questions. Please think step by step. When you're ready, please finish your answer with \"The answer is (X)\" where X is the correct letter choice. Please always include the parentheses around the letter choice."
        }]
        return messages_gpqa
    elif dataset_name == "custom_simple": # zero shots
        message_custom = [{
            "role":"system",
            "content":"You are a smart assistant in solving arithmetic questions. Please think step by step. If you think you're ready to output the answer, you can wrap your answer with \\boxed{}. Please follow this format. "
        }]
        return message_custom

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


def get_dataset_key(dataset_name):
    if dataset_name == "arc" or dataset_name == "ecqa" or dataset_name == "gsm8k" or dataset_name == "mmlu_pro" or dataset_name == "gsm8k_symbolic" or dataset_name == "mmlu":
        return "question"
    elif dataset_name == "math":
        return "problem"
    elif dataset_name == "proofwriter":
        return "context"
    elif dataset_name == "trivia_qa":
        return "question"
    elif dataset_name == "gpqa":
        return "question" # TODO: check the revision from Question to question
    elif dataset_name == "custom_simple":
        return "question"
    

def get_process_answer(dataset_name, data):
    if dataset_name == "gsm8k" or dataset_name == "gsm8k_symbolic":
        return data['answer']
    elif dataset_name == "math":
        return data["solution"]
    elif dataset_name == "trivia_qa":
        return data['answer']["normalized_value"]
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
        return str(data['answer'])

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
    elif dataset_name == "trivia_qa":
        return normalize_answer(prediction)
    
    # TODO: please double check the implementation. I've created test cases for them in test_re.py
    if dataset_name == "mmlu":  # extract the formatted answer
        regex1 = re.compile(r"answer is \(?\(([A-D])\)?\)")
        if (match := regex1.search(prediction)):
            return match.group(1).upper()
        regex2 = re.compile(r"\.*\[aA\]nswer:\s*\(([A-D])\)")
        if (match := regex2.search(prediction)):
            return match.group(1).upper()
        regex3 = re.compile(r"answer is \(?([A-D])\)?", re.IGNORECASE)
        if (match := regex3.search(prediction)):
            return match.group(1).upper()
        return "X"  # represent not found
    
    elif dataset_name == "mmlu_pro":  # extract the formatted answer
        regex1 = re.compile(r"answer is \(?\(([A-J])\)?\)")
        if (match := regex1.search(prediction)):
            return match.group(1).upper()
        regex2 = re.compile(r"\.*\[aA\]nswer:\s*\(([A-J])\)")
        if (match := regex2.search(prediction)):
            return match.group(1).upper()
        regex3 = re.compile(r"answer is \(?([A-J])\)?", re.IGNORECASE)
        if (match := regex3.search(prediction)):
            return match.group(1).upper()
        return "X"  # represent not found
    
    elif dataset_name == "gpqa":
        regex1 = re.compile(r"answer is \(?\(([A-D])\)?\)")
        if (match := regex1.search(prediction)):
            return match.group(1).upper()
        regex2 = re.compile(r"\.*\[aA\]nswer:\s*\(([A-D])\)")
        if (match := regex2.search(prediction)):
            return match.group(1).upper()
        regex3 = re.compile(r"answer is \(?([A-D])\)?", re.IGNORECASE)
        if (match := regex3.search(prediction)):
            return match.group(1).upper()
        return "X"  # represent not found

    elif dataset_name == "custom_simple" or "custom_hard":
        res = remove_boxed(last_boxed_only_string(prediction)) # TODO: please check if have time. Used the format for math questions
        return res

def flatten_list(new_messages):
    messages = []
    for message in new_messages:
        messages.append(message["role"] + ": " + message["content"])
    return "\n".join(messages)


def is_equivalent(dataset, item, data):
    if dataset == "math":
        try:
            a, b = normalize_final_answer(remove_boxed(last_boxed_only_string((item["full_response"][0])))), normalize_final_answer(remove_boxed(last_boxed_only_string(data["solution"])))
            if len(item["normalized_prediction"]) >= 1 and (is_equiv(a, b) or a.strip() == b.strip()):
                return True
            if len(item["normalized_prediction"]) >= 1 and item["normalized_prediction"][0] == item["normalized_answer"]:
                return True
        except:
            return False
        return False
    elif dataset == "arc" or dataset == "gsm8k" or dataset == "gpqa" or dataset == "mmlu_pro" or dataset == "gsm8k_symbolic" or dataset == "mmlu" or dataset == "custom_simple":
        if len(item["normalized_prediction"]) >= 1 and item["normalized_prediction"][0] == item["normalized_answer"]:
            return True
        else:
            return False
    elif dataset == "trivia_qa":
        if len(item["normalized_prediction"]) >= 1 and item["normalized_prediction"][0] in data['answer']['normalized_aliases']:
            return True
        else:
            return False
    

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
    elif dataset == "math" or dataset == "arc" or dataset == "trivia_qa" or dataset == "custom_simple":
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

def mask_answer_in_string(input_string, ground_truth):
    ground_truth_str = str(ground_truth)
    masked_string = re.sub(rf'\b{ground_truth_str}\b', '<answer masked>', input_string)
    return masked_string

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
    elif dataset == "math" or dataset == "custom_simple":
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
def generate_question(dataset, data): # please check do we need to add the original question once again at the end for completness? see example below
    # get previous is used for prompting the model while generate_question is used for record the question
    if dataset == "arc":
        question = data[get_dataset_key(dataset)] + "\n\nChoices: " + '\n'.join(data["choices"]["text"])
    elif dataset == "math" or dataset == "trivia_qa" or dataset == "gsm8k" or dataset == "custom_simple" or dataset == "custom_hard": # or dataset == "gpqa"
        question = data[get_dataset_key(dataset)]
    elif dataset == "mmlu":
        question = data[get_dataset_key(dataset)]  + "\nChoices: " + '\n'.join(data['choices'])
    elif dataset == "mmlu_pro":
        question = data[get_dataset_key(dataset)]  + "\nChoices: " + '\n'.join(data['options'])
    elif dataset == "gpqa":
        question = data[get_dataset_key(dataset)]  + "\nChoices: " + '\n'.join(data['options'])
        # example: question = data[get_dataset_key(dataset)]  + original question + "\nChoices: " + '\n'.join(data['options'])
    return question


