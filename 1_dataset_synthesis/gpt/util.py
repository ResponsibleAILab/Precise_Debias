def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
def save_file(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(data)