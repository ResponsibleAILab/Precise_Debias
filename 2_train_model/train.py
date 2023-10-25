import os
import json
import math
from time import time

from lib.util import save_file, open_file, get_learning_rate
from lib.server import start_infer_servers

from test_model import test_model
from initial_fine_tune import initial_fine_tune
from further_fine_tune import further_fine_tune

# Settings
lora_folder = 'model_servers/ethnicity_lora'
model_folder = 'model_servers/model'
training_data_file = 'data/fine_tune_data.json'
instruction_file_name = 'data/instruction.txt'
truth_file_name = 'data/bls_stats.json'
results_file_name = 'data/bias_results.json'
debug = True
persistant_servers = False
num_infer_servers = 1
target_bias = 0.02
max_runs = 22
initial_learning_rate = 3e-4
further_learning_rate = 7e-4

# Initialize logging
if not os.path.exists(truth_file_name):
    raise Exception(f"Could not find truth file {truth_file_name}")

def get_or_default(prop: str, obj: dict, default: any):
    if prop in obj:
        return obj[prop]
    return default

log_obj = { }
if os.path.exists(results_file_name):
    print('Found results file, loading data...')
    log_obj = json.loads(open_file(results_file_name))
    log_obj['target_bias'] = target_bias
    log_obj['initial_learning_rate'] = initial_learning_rate
    log_obj['further_learning_rate'] = further_learning_rate
else:
    print('Results file not found, starting over')
    log_obj = {
        "target_bias": target_bias,
        "initial_learning_rate": initial_learning_rate,
        "further_learning_rate": further_learning_rate,
        "target": json.loads(open_file(truth_file_name)),
        "iterations": 0,
    }
run = get_or_default('iterations', log_obj, 1)
max_runs += run
score_times = []
fine_tune_times = []
iteration_times = []
iteration_results = get_or_default('iteration_data', log_obj, [])
start_total_time = time()

print('\nTraining Configuration:')
for key in log_obj.keys():
    if key == 'iteration_data':
        continue
    print(f'{key}: {log_obj[key]}')
print(f'Max Runs: {max_runs}')

if not os.path.exists(model_folder):
    raise Exception(f'Model folder not found: {model_folder}')

# Get number of training examples
completions = json.loads(open_file(training_data_file))
num_training_examples = len(completions)


# Do initial training
if not os.path.exists(lora_folder):
    initial_fine_tune(training_data_file, instruction_file_name, model_folder, lora_folder, initial_learning_rate)

# Start Servers
infer_processes = []
infer_ports = []

if persistant_servers:
    (infer_processes, infer_ports) = start_infer_servers(num_infer_servers)

if run == 0:
    run = 1

try:
    while run < max_runs:
        start_score_time = time()
        [bias, percentages, results] = test_model(infer_ports, truth_file_name, instruction_file_name, debug)
        
        results += f'Run: {run}\n'
        results += f'Run time: {(time() - start_total_time) // 60:.2f}m\n'
        print(results)
        score_time = time() - start_score_time
        start_fine_tune_time = time()

        stop_loop = True
        for key in bias.keys():
            if math.fabs(bias[key]) > target_bias:
                stop_loop = False
        if not stop_loop and run < max_runs:
            further_fine_tune(training_data_file, instruction_file_name, model_folder, lora_folder, bias, further_learning_rate)

        learning_rates = { }
        for key in bias.keys():
            learning_rates[key] = get_learning_rate(further_learning_rate, bias[key], num_training_examples // len(bias.keys()))

        score_times.append(score_time)
        fine_tune_time = time() - start_fine_tune_time
        fine_tune_times.append(fine_tune_time)
        iteration_time = time() - start_score_time
        iteration_times.append(iteration_time)
        
        iteration_obj = {
            "run": run,
            "bias": bias,
            "percentages": percentages,
            "learning_rates": learning_rates,
            "result_str": results,
            "score_time": score_time,
            "fine_tune_time": fine_tune_time,
            "iteration_time": iteration_time
        }
        iteration_results.append(iteration_obj)
        log_obj['iteration_data'] = iteration_results
        log_obj['iterations'] = run
        log_obj['avg_score_time'] = sum(score_times) / run
        log_obj['avg_fine_tune_time'] = sum(fine_tune_times) / run
        log_obj['avg_iteration_time'] = sum(iteration_times) / run
        log_obj['total_time'] = time() - start_total_time
        save_file(results_file_name, json.dumps(log_obj))
        run += 1
        if stop_loop:
            print('Training Complete')
            break
finally:
    for p in infer_processes:
        p.terminate()
