import yaml
import json
import os
import logging
import shutil
import pandas as pd

def read_params(config_path: str) -> dict:
    try:
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        # logging.info(f"read parameters")
        return config
    except Exception as e:
        raise e

def clean_prev_dirs_if_exists(dir_path_list: list):
    try:
        for dir_path in dir_path_list:
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
        # logging.info(f"cleaned existing artifacts directory at {dir_path}")
    except Exception as e:
        raise e

def create_dirs(dirs: list):
    try:
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            # logging.info(f"created directory at {dir_path}")
    except Exception as e:
        raise e

def move_files(file_path_list: list, dir: str, remove_file=False):
    try:
        for file in file_path_list:
            shutil.copy(file, dir)
            if remove_file:
                os.remove(file)
    except Exception as e:
        raise e

def get_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise e