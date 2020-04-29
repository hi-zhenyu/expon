import os
import json

def jsonstr2dict(json_str):
    return json.loads(json_str)

def dict2jsonstr(dic, is_format=True):
    if is_format:
        return json.dumps(dic, sort_keys=False, indent=4, separators=(',', ': '))
    return json.dumps(dic)

def raise_exception(str):
    raise Exception(str)

def get_root_path():
    return os.getcwd()

def check_file(file_path):
    return os.path.isfile(file_path)
    
def check_dir(dir_path):
    return os.path.isdir(dir_path)

def create_dir(dir_path):
    if not check_dir(dir_path):
        os.makedirs(dir_path)

def create_dir_from_file(file_path):
    file_dir = os.path.split(file_path)[0]
    create_dir(file_dir)