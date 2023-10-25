import json
import socket
from time import time, sleep

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

ethnicities = json.loads(open_file('data/bls_stats.json')).keys()

def get_ethnicity(prompt: str):
    l_prompt = prompt.lower()
    for ethnicity in ethnicities:
        if ethnicity in l_prompt:
            return ethnicity
    return None
    
def save_file(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(data)

def get_alpaca_prompt(input: str, instruction: str) -> str:
    return f"""
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.


### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

def print_run_time(idx: int, total: int, start_time: float):
    run_time = time() - start_time
    print(f'Current Run time #{idx} of {total}: {run_time / 60 // 60}h {(run_time // 60) % 60}m {run_time % 60:.2f}s')

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
        
def get_learning_rate(initial_learning_rate: float, bias: float, num_items: int) -> float:
    return initial_learning_rate * bias / num_items