import re
import string
import random
import datasets
import asyncio
import aiohttp
import numpy as np
import json
from datetime import datetime
from math_utils import normalize_final_answer, remove_boxed, last_boxed_only_string, is_equiv


gsm8k_list_of_end_prompts = ['the answer is', 'The correct answer is', 'The answer is', 'The answer is indeed', 'the correct answer is indeed']

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string



def extract_and_convert_number_real(text):
    text = str(text)
    # deal with 2 + 3 = 5
    if '=' in text:
        text = text.split('=')[1].strip()
    # remove end dot
    if text.endswith('.'):
        text = text[:-1]
    # deal with 13.00 and 13
    if '.' in text:
        text = text.split('.')[0]
    pattern = re.compile(r'(\-?[0-9\.,]+)')
    match = pattern.search(text)
    if match:
        # Remove commas from the number, if any, and convert to float
        number_str = match.group().replace(',', '')
        return number_str
    else:
        return text

    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


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
        return data['problem']
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
    

def get_process_answer(dataset_name, data):
    if dataset_name == "gsm8k":
        return data['answer']
    elif dataset_name == "math":
        return data["solution"]
    elif dataset_name == "trivia_qa":
        return data['answer']["normalized_value"]


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
    

def flatten_list(new_messages):
    messages = []
    for message in new_messages:
        messages.append(message["role"] + ": " + message["content"])
    return "\n".join(messages)
        

async def call_vllm_server(agent_model, new_messages, temperature, n, tokenizer, base_url, ports, cache, type=None, dataset=None, round=None):
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
        agent_response = agent_response['choices'][0]['message']['content']
    except:
        agent_response = ""
    # Store responses
    if type is None or dataset is None or round is None:
        raise ValueError("Type or dataset is None")
    cache.store(
        prompt=flatten_list(new_messages),
        response=agent_response,
        model=agent_model,
        temperature=temperature,
        type=type,
        dataset=dataset,
        round=round
    )
    
    return agent_response   

def mask_answer_in_string(input_string, ground_truth):
    ground_truth_str = str(ground_truth)
    masked_string = re.sub(rf'\b{ground_truth_str}\b', '<answer masked>', input_string)
    return masked_string

def check_if_ground_truth_exists(input_string, ground_truth):
    # return True if ground truth exists
    ground_truth_str = str(ground_truth)
    match = re.search(rf'\b{ground_truth_str}\b', input_string)
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
    return normalized_prediction_list


def generate_question(dataset, data):
    if dataset == "arc":
        question = data[get_dataset_key(dataset)] + "\n\nChoices: " + '\n'.join(data["choices"]["text"])
    elif dataset == "math" or dataset == "trivia_qa" or dataset == "gsm8k" or dataset == "gpqa":
        question = data[get_dataset_key(dataset)]
    return question


def is_equivalent(dataset, item, data):
    if dataset == "math":
        try:
            a, b = normalize_final_answer(remove_boxed(last_boxed_only_string((item["full_response"][0])))), normalize_final_answer(remove_boxed(last_boxed_only_string(data["solution"])))
            if len(item["normalized_prediction"]) >= 1 and (is_equiv(a, b) or a.strip() == b.strip()):
                return True
        except:
            return False
    
