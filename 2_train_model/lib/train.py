import os
import json
import time
import torch
from typing import List
from datasets import Dataset
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling 

from lib.util import open_file

def train_model(model_folder: str, lora_folder: str, dataset: List[str], learn_rate: float, epochs: int):
    start_time = time.time()
    train_config_file = 'lora_train_config.json'
    train_config = {
        "micro_batch_size": 1,
        "batch_size": 1
    }

    if os.path.exists(train_config_file):
        train_config = json.loads(open_file(train_config_file))

    print('Loading LLM')
    # load llama model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_folder,
        load_in_8bit=True,
    )
    model.enable_input_require_grads()
    model_load_time = time.time()
    print(f'Loaded LLM in {model_load_time - start_time:.2f}s')

    print('Loading LoRa model')
    new_model = not os.path.exists(lora_folder)
    if not new_model:
        # retrieve and apply pretrained lora
        model = PeftModel.from_pretrained(
            model=model,
            model_id=lora_folder,
            adapter_name='ethnicity_lora',
            is_trainable=True
        )
    else:
        print(f'Lora folder ({lora_folder}) does not exist, starting over')
        lora_config = LoraConfig(
            task_type='CAUSAL_LM',
            r=512,
            lora_alpha=1024,
            inference_mode=False,
            init_lora_weights=True,
            bias='none',
            fan_in_fan_out=False,
            lora_dropout=0.05,
            peft_type='LORA',
            target_modules=[
                'q_proj',
                'v_proj'
            ]
        )
        model = get_peft_model(model, lora_config)
    peft_load_time = time.time()
    print(f'Loaded LORA in {peft_load_time - model_load_time:.2f}s')

    print('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer_load_time = time.time()
    print(f'Loaded tokenizer in {tokenizer_load_time - peft_load_time:.2f}s')

    def tokenize(data: str):
        cutoff = 256
        input_ids = tokenizer.encode(data, truncation=True, max_length=cutoff)
        input_ids = [tokenizer.pad_token_id] * (cutoff - len(input_ids)) + input_ids
        labels = [1] * len(input_ids)
        input_ids = torch.tensor(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(tokenizer.pad_token_id)
        }
    
    gradient_steps = train_config['batch_size'] // train_config['micro_batch_size']
    trainer = Trainer(
        model=model,
        train_dataset=Dataset.from_list([tokenize(x) for x in dataset]),
        args=TrainingArguments(
            per_device_train_batch_size=train_config['micro_batch_size'],
            gradient_accumulation_steps=gradient_steps,
            num_train_epochs=epochs,
            learning_rate=learn_rate,
            fp16=True,
            optim='adamw_torch',
            evaluation_strategy='no',
            lr_scheduler_type='constant',
            output_dir='output'
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    try:
        trainer.train()
    finally:
        save_folder = lora_folder
        if not new_model:
            dirs = lora_folder.split('/')
            save_folder = "/".join(dirs[:len(dirs)-1])
        print(f'Saving LoRa to {save_folder}')
        model.save_pretrained(save_folder)