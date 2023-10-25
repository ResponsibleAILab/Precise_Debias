import requests
from typing import List

def llm_complete_async(prompt: str, top_k: int, stop_token: str, num_tokens: int, ports: List[int]) -> List[str]:
    for port in ports:
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
    completions = []
    for port in ports:
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
        completions.append(completion)
    return completions