import json

from lib.train import train_model
from lib.util import open_file, get_alpaca_prompt, get_ethnicity, get_learning_rate

def further_fine_tune(training_file_name: str, instruction_file_name: str, model_folder: str, lora_folder: str, bias: dict, initial_learning_rate: float):
    instruction = open_file(instruction_file_name)
    completions = json.loads(open_file(training_file_name))
    idx = 0
    print('Starting Further Fine tune')
    training_runs = { }
    for completion in completions:
        idx += 1
        ethnicity = get_ethnicity(completion['output'])
        if ethnicity is None:
            continue
        if ethnicity not in training_runs:
            training_runs[ethnicity] = []
        training_runs[ethnicity].append(
            f"{get_alpaca_prompt(completion['input'], instruction)}{completion['output']}"
        )
    for key in training_runs.keys():
        if bias[key] < 0:
            continue
        examples = training_runs[key]
        learning_rate = get_learning_rate(initial_learning_rate, bias[key], len(examples))
        print(f'Training {learning_rate} for {key} with {len(examples)} ({len(examples) / len(completions) * 100:.2f}%) items')
        train_model(model_folder, lora_folder, examples, learning_rate, 1)