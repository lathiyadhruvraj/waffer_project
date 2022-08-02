import os
import boto3
from utils.common_utils import read_params, clean_prev_dirs_if_exists, create_dirs
import argparse
from dotenv import load_dotenv


def get_data_from_bucket(config):
    # config = read_params(config_path)

    root_dir = config['artifacts']['artifacts_dir']
    s3_bucket_name = config['s3_data_source']['bucket_name']
    s3_root_folder_prefix = config['s3_data_source']['s3_root_folder_prefix']
    s3_folder_list = config['s3_data_source']['s3_folder_list']
    bucket_folder_name = config['s3_data_source']['bucket_folder_name']

    try:
        clean_prev_dirs_if_exists([os.path.join(root_dir, bucket_folder_name)])
        create_dirs([os.path.join(root_dir, bucket_folder_name)])
        my_bucket = s3.Bucket(s3_bucket_name)
        print("Downloading Files from s3 bucket")
        i = 1
        for file in my_bucket.objects.filter(Prefix=s3_root_folder_prefix):
            if any(s in file.key for s in s3_folder_list): 
                path, filename = os.path.split(file.key)
                os.makedirs(os.path.join(root_dir, path), exist_ok=True)
                print(i, "Downloading ===> ", file.key)
                my_bucket.download_file(file.key, os.path.join(root_dir, path, filename))
                i += 1
        print(" Downloading Complete")
    except Exception as err:
        print(err)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        config_path=parsed_args.config
        config = read_params(config_path)

        load_dotenv()

        Access_key_ID = os.getenv('Access_key_ID')
        Secret_access_key = os.getenv('Secret_access_key')

        s3 = boto3.resource("s3", region_name=config['s3_data_source']['region_name'], 
                                aws_access_key_id=Access_key_ID, 
                                aws_secret_access_key=Secret_access_key)
        
        data = get_data_from_bucket(config=config)
    except Exception as err:
        print(err)