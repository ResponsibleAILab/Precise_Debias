import os
import json
from lib.train import train_model
from lib.util import open_file, get_alpaca_prompt

def initial_fine_tune(fine_tune_data_file_name: str, instruction_file_name: str, model_folder: str, lora_folder: str, initial_learning_rate: float):
    if not os.path.exists(fine_tune_data_file_name):
        raise Exception(f"Could not finde fine tune data file: {fine_tune_data_file_name}")
    print('Starting initial fine tune')
    completions = json.loads(open_file(fine_tune_data_file_name))
    instruction = open_file(instruction_file_name)
    training_examples = []
    for item in completions:
        inp = get_alpaca_prompt(item['input'], instruction)
        out = item['output']
        training_examples.append(f"{inp}{out}")
    print('Example:')
    print(training_examples[0])
    train_model(model_folder, lora_folder, training_examples, initial_learning_rate, 1)
    print('Done with initial fine tune')
