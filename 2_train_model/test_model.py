import os
import json
import requests
from time import sleep, time
from typing import List, Tuple

from lib.api import llm_complete_async
from lib.server import start_infer_servers
from lib.util import open_file, get_alpaca_prompt, print_run_time, get_ethnicity

def get_llm_completions(num_responses: int, prompt: str, ports: List[int], debug: bool) -> List[str]:
    if debug:
        print(f'Running LLM inference {num_responses} times')

    outputs = []
    for _ in range(num_responses):
        new_outputs = llm_complete_async(prompt, 4, '\n##', 10, ports)
        if new_outputs == None:
            raise Exception('Error in response')
        for response_txt in new_outputs:
            outputs.append(response_txt)
            if debug:
                print(f"Response: \n{prompt + response_txt}")
    return outputs

def infer_llm(ports, instruction_file_name: str, debug: bool) -> List[str]:
    num_responses = 10
    num_inferences = num_responses // len(ports)
    
    start_run_time = time()
    
    instruction = open_file(instruction_file_name)
    data = json.loads(open_file('data/evaluation_data.json'))
    completions = []
    inputs = []
    idx = 0
    # eliminate duplicates
    for item in data:
        if item['input'] in inputs:
            continue
        inputs.append(item['input'])
    print(f'Inferring {len(inputs)} inputs')
    for inp in inputs:
        try:
            start_infer_time = time()
            prompt = get_alpaca_prompt(inp, instruction)
            outputs = get_llm_completions(num_inferences, prompt, ports, debug)
            for res in outputs:
                completions.append(res)
            if debug:
                print(f'LLM infered in {time() - start_infer_time:.2f}s')
            print_run_time(idx, len(inputs), start_run_time)
            idx += 1
        except Exception as e:
            print(e)
            print('Error occurred, resting 10 seconds...')
            sleep(10)
            continue
    return completions

def get_totals(outputs: List[str]) -> Tuple[dict, int]:
    totals = { }
    num_found = 0
    for output in outputs:
        ethnicity = get_ethnicity(output)
        if ethnicity is None:
            continue
        if ethnicity not in totals:
            totals[ethnicity] = 0
        totals[ethnicity] += 1
        num_found += 1
    return (totals, num_found)

def get_percentages(totals: dict, num_found: int):
    percentages = { }
    for key in totals.keys():
        percentages[key] = totals[key] / num_found
    return percentages

def get_bias(percentages: dict, ground_truth: dict, debug=False) -> dict:
    bias = { }
    for key in percentages.keys():
        bias[key] = ground_truth[key] - percentages[key]
    return bias

def get_results(num_found: int, inferences: List[str], totals: dict, bias: dict, percentages: dict) -> str:
    results = f'Matched: {num_found} of {len(inferences)} ({num_found/len(inferences)*100:.2f}%)\n'
    results += 'Totals:\n'
    for key in percentages.keys():
        results += f'{key}: {percentages[key] * 100:.2f}%\n'

    results += f'\n\nBias: (lower is better)\n'
    for key in bias.keys():
        results += f'{key}: {bias[key]}\n'
    return results

def test_model(ports: List[int], truth_file: str = '', instruction_file: str = '', debug: bool = False):
    if instruction_file == '':
        raise Exception('Instruction file not specified')
    if not os.path.exists(instruction_file):
        raise Exception('Could not find instruction file')
    if truth_file == '':
        raise Exception('Truth file not specified')
    if not os.path.exists(truth_file):
        raise Exception('Could not find truth file')
    processes = []
    if len(ports) > 0:
        # Reload LoRa file on servers
        for port in ports:
            requests.post(f'http://localhost:{port}/reload', json={ })
    else:
        (processes, ports) = start_infer_servers(1)
    try:
        ground_truth = json.loads(open_file(truth_file))

        inferences = infer_llm(ports, instruction_file, debug)
        (totals, num_found) = get_totals(inferences)
        percentages = get_percentages(totals, num_found)
        bias = get_bias(percentages, ground_truth)
        
        results = get_results(num_found, inferences, totals, bias, percentages)
    finally:
        if len(processes) > 0:
            print('Terminating LLM servers')
            for port in ports:
                try:
                    requests.post(f'http://localhost:{port}/stop')
                except:
                    continue
            pids = [p.pid for p in processes]
            for p in processes:
                p.terminate()
            # Ensure termination
            for pid in pids:
                os.system(f'kill {pid}')

    return [bias, percentages, results]
