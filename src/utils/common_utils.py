import yaml
import json
import os
import logging
import shutil


def read_params(config_path: str) -> dict:
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    # logging.info(f"read parameters")
    return config

def clean_prev_dirs_if_exists(dir_path: str):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    # logging.info(f"cleaned existing artifacts directory at {dir_path}")


def create_dir(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        # logging.info(f"created directory at {dir_path}")