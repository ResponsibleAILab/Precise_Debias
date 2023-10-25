import re
from time import time

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
def save_file(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(data)

def concatList(s, l, c):
    for i in l:
        s += c + " " + i
    return s

def reformatPrompts(list):
    list = concatList('', re.findall("\d\..*\n", list + "\n"), '')
    return re.sub("\d\. ", "- ", list)

def print_run_time(idx: int, total: int, start_time: float):
    run_time = time() - start_time
    print(f'Current Run time #{idx} of {total}: {run_time / 60 // 60}h {(run_time // 60) % 60}m {run_time % 60:.2f}s')