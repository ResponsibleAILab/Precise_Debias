import os
import json
from time import time
from typing import List

import openai

from gpt.lib import GptCompletion
from lib.util import open_file, save_file, concatList, reformatPrompts, print_run_time

if not os.path.exists('config.json'):
    raise Exception('config.json not found')

config = json.loads(open_file('config.json'))

if 'openai_key' not in config:
    raise Exception('\'openai_key\' property not found in config.json')

openai.api_key = config['openai_key']

def make_initial_prompt(examples: List[str]) -> str:
    example_string = concatList('', examples, '\n- ')
    return \
        f"Write a few prompts for a text to image model that describes a person in a specific occupation. \n\
        Try to include these ideas in the prompts: diversity, specificity, imagination, detail, and emotion.\n\
        Use language that paints a picture and describes a story.\n\
        Always include the ethnicity of the person in the prompt \n \
        Example Prompts:{example_string}\n\nNew Prompts:"

def generate_prompts(examples: List[str], gptComp: GptCompletion) -> List[str]:
    remove_end = { "50256": -100 }
    res = gptComp.complete(make_initial_prompt(examples), {
        "max_tokens": 200,
        "temperature": 1,
        "logit_bias": remove_end
    })
    return res

def bulk_generate_prompts(max_tokens: int, gptComp: GptCompletion, save_file_name: str) -> List[str]:
    ret_val = []
    save_interval = 10
    start_time = time()
    num_calls = 0
    while gptComp.total_tokens < max_tokens:
        num_calls += 1
        prompts = generate_prompts(examples, gptComp)
        for prompt in prompts.split('\n-'):
            if prompt == ' ':
                continue
            prompt = prompt.replace('- ', '')
            print(prompt)
            ret_val.append(prompt)
        if num_calls % save_interval == 0:
            save_file(save_file_name, json.dumps(ret_val))
        print('\n\n')
        print_run_time(num_calls, 'inf', start_time)
        print(f'Total Tokens: {gptComp.total_tokens}')
        print('\n\n')
    return ret_val

gptComp = GptCompletion()

examples = [
    "A male nurse of asian descent is checking the vital signs of a patient in a hospital",
    "A female nurse of african descent is administering medication to a patient in a hospital bed",
    "A male nurse of hispanic heritage is comforting a child patient in a pediatric unit",
    "An African American woman confidently leading a boardroom meeting, with a diverse team of executives listening attentively and contributing their ideas.",
    "A young Asian male CEO giving a motivational speech to a large audience, emphasizing the importance of inclusivity and diversity in the workplace.",
    "A Latina CEO addressing her employees during a company-wide meeting, highlighting the achievements of individuals with disabilities and promoting an inclusive work environment.",
    "An older white male CEO mentoring a young female employee, sharing his wisdom and guiding her through her professional development."
]

save_file_name = 'data/synthetic_prompts.json'
p = bulk_generate_prompts(50000, gptComp, save_file_name)
save_file(save_file_name, json.dumps(p))