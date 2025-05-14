from scienceworld import ScienceWorldEnv 
import yaml
from argparse import ArgumentParser
import requests
import json
import copy
from jinja2 import Environment, FileSystemLoader
import traceback
from pydantic import BaseModel
from openai import OpenAI

class AgentResponse(BaseModel):
    reasoning: str
    action: str
    finished: bool

class LocalVLLMAgent(object):
    def __init__(
        self,
        endpoint,
        k=1
    ):
        self.k = k
        self.client = OpenAI(
            base_url=endpoint,
            api_key="-",
        )
        self.json_schema = AgentResponse.model_json_schema()
    
    def __call__(self, messages, generation_params):
        data = copy.deepcopy(generation_params)
        data['extra_body'] = {"guided_json": self.json_schema}
        res = self.client.chat.completions.create(
            messages=messages,
            **data
        )
        try:
            return res.choices[0].message.content
        except Exception as e: #e
            return ''
    
def main():
    parser = ArgumentParser(description="Run ScienceWorld Environment")
    parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    # Load the configuration file (yaml)
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    agent = LocalVLLMAgent(
        endpoint=f"{config['base_url']}:{config['ports']}/v1",
        k=1
    )

    llm_jinja_env = Environment(
        loader=FileSystemLoader(config['prompts']['dir']),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    
    env = ScienceWorldEnv("", config['scienceworld_path'], envStepLimit=config['turn_budget'])
    taskNames = env.get_task_names()

    tasks = range(30)
    variations = range(10)
    for taskIdx in tasks:
        for variation in variations:
            # Create the environment
            taskName = taskNames[taskIdx]
            env.load(taskName, variation, "", generateGoldPath=True)
            
            # Reset the environment to start a new episode
            print(f"*" * 80)
            env.reset()

            # Initialize the messages for the agent
            task_description = env.get_task_description()
            system_data = {'task_description': task_description}
            system_prompt = llm_jinja_env.get_template(config['prompts']['system']).render(system_data)
            messages = [{'role': 'system', 'content': system_prompt}]
            print("task description:", task_description)

            # Run a single episode
            done = False
            reasoning = ""
            action = "look around"
            record = {
                'task_name': taskName,
                'variation': variation,
                'reasoning': [],
                'actions': [],
                'scores': [],
                'rewards': [],
                'status': [],
            }
            count = 0
            while not done and count < config['turn_budget']:
                count += 1
                # breakpoint()

                # Take a step in the environment using the sampled action
                observation, reward, done, info = env.step(action)
                record['rewards'].append(reward); record['status'].append(done); record['actions'].append(action); record['reasoning'].append(reasoning); record['scores'].append(info['score'])
                if done: 
                    print(info['score'])
                    break
                print(f"{action}: {reasoning}", end="\r\n"); print(f"score: {info['score']}", end="\r")

                # Add the current observation to the messages
                user_data = {'observation': observation, 'possible_actions': env.get_possible_actions(), 'possible_objects': env.get_possible_objects()}
                user_prompt = llm_jinja_env.get_template(config['prompts']['user']).render(user_data)
                messages.append({'role': 'user', 'content': user_prompt})

                # Sample a random action from the action space
                response = agent(messages, config['generation_params'])
                messages.append({'role': 'assistant', 'content': response})

                # Parse the response
                try:
                    response = json.loads(response)
                    action = response['action']
                    done = response['finished']
                    reasoning = response['reasoning']
                except Exception as e:
                    traceback.print_exc()
                    print(repr(e))
                    breakpoint()
            # else: # debug
            #     print(response['finished'])
            #     print(done)
            #     print(count)
            #     print(config['turn_budget'])
            


if __name__ == "__main__":
    main()