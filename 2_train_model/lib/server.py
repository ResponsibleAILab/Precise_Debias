import os
import multiprocessing
from typing import List, Tuple

from lib.util import wait_for_port

def start_single_infer_server() -> Tuple[List[multiprocessing.Process], List[int]]:
    print('Starting LLM Servers')
    process = multiprocessing.Process(target=create_infer_server, args=(1, ))
    process.start()
    
    wait_for_port(5011)
    return ([process], [5011])

def create_infer_server(port: int, gpu_num: int):
    os.system(f'python model_servers/infer_llm_server.py {port} {gpu_num} model_servers/model model_servers/ethnicity_lora')

def start_infer_servers(num_servers: int) -> Tuple[List[multiprocessing.Process], List[int]]:
    print('Starting LLM Servers')
    ports = [5010 + i for i in range(1,num_servers+1)]
    processes = []
    for port in ports:
        process = multiprocessing.Process(target=create_infer_server, args=(port, port - 5010, ))
        process.start()
        processes.append(process)
    
    for port in ports:
        wait_for_port(port)
    return (processes, ports)