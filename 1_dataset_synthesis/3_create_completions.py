import json

from lib.util import open_file, save_file

ethnicities = [
    "Black male",
    "Black female", 
    "White male",
    "White female", 
    "Hispanic male", 
    "Hispanic female", 
    "Asian male",
    "Asian female"
]

prompts_file_name = 'data/replaced_prompts.json'
prompts = json.loads(open_file(prompts_file_name))
print(f'Loaded {len(prompts)} prompts')

pairs = []
for completion in prompts:
    # copy string for modification
    prompt = completion.replace('[ETHNICITY] ', '')
    
    for ethnicity in ethnicities:
        output = completion.replace('[ETHNICITY]', ethnicity)
        # Alpaca format
        pairs.append({
            "input": prompt,
            "output": output + '\n###\n'
        })

portion_train = 0.8
evaluation_cutoff = int(len(pairs) * portion_train)

train_pairs = pairs[:evaluation_cutoff]
evaluation_pairs = pairs[evaluation_cutoff:]

print(f'Created {len(pairs)} prompt/completion pairs')
print(f'{len(train_pairs)} Training pairs')
print(f'{len(evaluation_pairs)} Evaluation pairs')
save_file('data/fine_tune_data.json', json.dumps(train_pairs))
save_file('data/evaluation_data.json', json.dumps(evaluation_pairs))