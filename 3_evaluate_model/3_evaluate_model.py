import os
import sys
import json
import time
import uuid
import base64
import random
import socket
import requests
import multiprocessing

from time import sleep
from typing import List, Tuple

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def check_port(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    opened = result == 0
    sock.close()
    return opened

def wait_for_port(port: int):
    while True:
        sleep(10)
        if check_port(port):
            return

def create_infer_server(port: int, gpu_num: int):
    os.system(f'python model_servers/infer_llm_server.py {port} {gpu_num} model_servers/model model_servers/lora')

def start_single_infer_server(port: int, gpu_num: int) -> Tuple[List[multiprocessing.Process], List[int]]:
    print('Starting LLM Server')
    process = multiprocessing.Process(target=create_infer_server, args=(port, gpu_num, ))
    process.start()
    wait_for_port(port)
    return process

def llm_complete(prompt: str, top_k: int, stop_token: str, num_tokens: int, port: int) -> str:
    res = requests.post(f'http://localhost:{port}/complete-async', json={
        'prompt': prompt,
        'top_k': top_k,
        'num_tokens': num_tokens
    })
    if res.status_code != 200:
        print(res.content)
        return None
    data = res.json()
    if 'error' in data:
        print(data['error'])
        return None
    
    res = requests.post(f'http://localhost:{port}/get-completions', json={ })
    if res.status_code != 200:
        print(res.content)
        return None
    data = res.json()
    if 'error' in data:
        print(data['error'])
        return None
    completion = data['completion'].split(prompt)[1]
    if stop_token in completion:
        completion = completion.split(stop_token)[0]
    return completion

def get_alpaca_prompt(input):
    return f"""
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.


### Instruction:
Expand this text to image prompt to be more diverse and inclusive

### Input:
{input}

### Response:
"""

def save_encoded_image(b64_image: str, prompt: str):
    file_name = str(uuid.uuid3(uuid.NAMESPACE_URL, str(random.random()))) + '.png'
    with open(f'images/{file_name}', "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))
    with open(f'images/{file_name}_prompt.txt', 'w') as f:
        f.write(prompt)

def get_sd_img(prompt, neg_prompt, num_images, min_images = 5, initial_images = [], fail_count = 0):
    if not os.path.exists('images'):
        os.mkdir('images')

    images = initial_images
    batch_size = 5
    iterations = num_images // batch_size
    for _ in range(0, iterations):
        print(f'Requesting SD for {batch_size} images')
        res = requests.post(f'{sd_ip}/sdapi/v1/txt2img', json={
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "steps": 10,
            "sampling_method": "UniPC",
            "batch_size": batch_size
        })
        if res.status_code != 200:
            print("Error with sd api")
            return None
        for i in res.json()['images']:
            images.append(i)
            save_encoded_image(i, prompt)

    if fail_count > 5:
        return images

    if len(images) < min_images:
        print('Could not get reccomended amount of images, trying again')
        return get_sd_img(prompt, neg_prompt, num_images, min_images, images, fail_count + 1)
    return images

# number of completions to generate
num_responses = 10
# Number of classes in training data
num_classes = 4
num_images_per_response = 5
llm_port = 5011
gpu_num = 1
eval_file = 'data/evaluation_data.json'

# Prompt to send to the llm
prompt = 'A nurse is caring for a premature baby in an incubator in the neonatal unit'
# negative prompt for stable diffusion
negative_prompt = "billboard, logo, words, word"

# Example: python 3_evaluate_model.py https://072fcca016ed96f7de.gradio.live
if len(sys.argv) < 2:
    raise Exception('Usage: python 3_evaluate_model.py [stable_diffusion_address]')

if not os.path.exists(eval_file):
    raise Exception(f'Evaluation file {eval_file} does not exist')

eval_data = json.loads(open_file(eval_file))

# stable diffusion api address
sd_ip = sys.argv[1]

def inference(infer,  input, start_time):
    print('Running LLM inference')
    prompt = get_alpaca_prompt(input)
    responses = []
    for _ in range(0, num_responses):
        response_txt = llm_complete(prompt, num_classes, '\n##', 50, llm_port) if infer else input
        print(f"Response: {response_txt}")
        responses.append(response_txt)
    llm_infer_time = time.time()
    print(f'LLM infered in {llm_infer_time - start_time:.2f}s')
    print(f'Created {len(responses)} responses')

    print('Sending Stable diffusion api request for expanded')
    all_imgs = []
    idx = 0
    for response in responses:
        start_sd_time = time.time()
        imgs = get_sd_img(response, negative_prompt, 5, True)[:5]
        print(f'Recieved batch {idx} of {len(responses)} - {time.time() - start_sd_time:.2f}s')
        idx += 1
        for img in imgs:
            all_imgs.append(img)
    sd_exp_time = time.time()
    print(f'SD infered in {sd_exp_time - llm_infer_time:.2f}s')
    return all_imgs

start_single_infer_server(llm_port, gpu_num)
start_run_time = time.time()


prompts = []
for item in eval_data:
    if item['input'] in prompts:
        continue
    prompts.append(item['input'])

print(f'Running inference on {len(prompts)} prompts')
for prompt in prompts:
    inference(True, prompt, start_run_time)
    print(f'Total inference time: {time.time() - start_run_time:.2f}s')
try:
    requests.post(f'http://localhost:{llm_port}/stop')
except:
    print('LLM Server stopped')