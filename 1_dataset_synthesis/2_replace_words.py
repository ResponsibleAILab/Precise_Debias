import os
import json

from lib.util import open_file, save_file

prompt_file_name = 'data/synthetic_prompts.json'
if not os.path.exists(prompt_file_name):
	raise Exception(f'Prompt list file \"{prompt_file_name}\" not found')

ethnicity_words_file_name = 'data/ethnicity_words.json'
if not os.path.exists(ethnicity_words_file_name):
	raise Exception(f'Ethnicity words file \"{ethnicity_words_file_name}\" not found')

ethnicity_words = json.loads(open_file(ethnicity_words_file_name))
prompts = json.loads(open_file(prompt_file_name))

print(f'Loaded {len(prompts)} prompts')

new_prompts = []
for prompt in prompts:
	replaced = False
	for wd in prompt.split(' '):
		if wd.lower() in ethnicity_words:
			prompt = prompt.replace(wd, '[ETHNICITY]')
			replaced = True
	if replaced and 'of [ETHNICITY]' not in prompt:
		idx = prompt.find('[ETHNICITY]')
		if idx == -1:
			continue
		prompt = prompt.replace('[ETHNICITY]', '')
		prompt = f'{prompt[:idx]}[ETHNICITY] {prompt[idx:]}'
		prompt = prompt.replace('   ', ' ').replace('  ', ' ')
		if prompt[0] == ' ':
			prompt = prompt[1:]
		new_prompts.append(prompt)

print(f'Saved {len(new_prompts)} prompts')

replaced_prompts_file_name = 'data/replaced_prompts.json'
save_file(replaced_prompts_file_name, json.dumps(new_prompts))