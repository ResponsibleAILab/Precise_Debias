import os
import sys
import time
import threading

import torch
from peft import PeftModel
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
app.debug = False

os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"

start_time = time.time()

if len(sys.argv) < 5:
    raise Exception('Usage: python infer_llm_server.py [PORT] [GPU_NUM] [model_folder] [lora_folder]')

port = sys.argv[1]
gpu_num = sys.argv[2]
model_folder = sys.argv[3]
lora_folder = sys.argv[4]

print(f'Starting server on port: {port}')
print(f'GPU Num: {gpu_num}')

if not os.path.exists(model_folder):
    raise Exception(f'Model folder {model_folder} does not exist')

if not os.path.exists(lora_folder):
    raise Exception(f'LoRa folder {lora_folder} does not exist')

print('Loading LLM')
# Load LLM
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_folder,
    # load_in_8bit=True # can only be used on primary GPU
).to(f'cuda:{gpu_num}')
model_load_time = time.time()
print(f'Loaded LLM in {model_load_time - start_time:.2f}s')

print('Loading LoRa model')
# retrieve and apply pretrained lora
model = PeftModel.from_pretrained(
    model=model,
    model_id=lora_folder,
    adapter_name='ethnicity_lora'
).to(f'cuda:{gpu_num}')
peft_load_time = time.time()
print(f'Loaded LoRa in {peft_load_time - model_load_time:.2f}s')

print('Loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained(model_folder)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer_load_time = time.time()
print(f'Loaded tokenizer in {tokenizer_load_time - peft_load_time:.2f}s')

completions_cache = []
completions_lock = threading.Lock()

def complete_async(prompt: str, top_k: int, num_tokens: int) -> str:
    with completions_lock:
        global completions_cache
        global tokenizer
        global model
        
        completions_cache = []
        encoded_input = tokenizer(prompt, return_tensors="pt").to(f'cuda:{gpu_num}')
        with torch.no_grad():
            output = model.generate(
                input_ids=encoded_input['input_ids'], 
                max_new_tokens=num_tokens, 
                top_k=top_k, 
                do_sample=True, # Very Important, entirely deterministic without
                temperature=1
            )
            completion = tokenizer.batch_decode(output.detach().cpu().numpy(), skip_special_tokens=True)[0]
            completions_cache.append(completion)

@app.route('/complete-async', methods=['POST'])
def process_async():
    data = request.json
    if 'prompt' not in data:
        return jsonify({ 'error': 'Prompt not found in request.'}), 200

    if 'top_k' not in data:
        return jsonify({ 'error': 'top_k not found in request.'}), 200

    num_tokens = 30
    if 'num_tokens' in data:
        num_tokens = data['num_tokens']

    thread = threading.Thread(target=complete_async, args=(data['prompt'], data['top_k'], num_tokens, ))
    thread.start()

    return jsonify({ 'started': True }), 200

@app.route('/get-completions', methods=['POST'])
def get_completions():
    with completions_lock:
        global completions_cache
        if len(completions_cache) == 0:
            return jsonify({ 'error': 'No inference started'}), 200
        
        return jsonify({ 'completion': completions_cache[0] }), 200

@app.route('/stop', methods=['POST'])
def stop_server():
    try:
        pid = os.getpid()
        os.system(f'kill {pid}')
    except:
        return ''

if __name__ == '__main__':
    app.run(port=port)