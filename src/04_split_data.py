from sklearn.model_selection import train_test_split
import argparse
import os 
from utils.common_utils import read_params, create_dirs, clean_prev_dirs_if_exists, get_data
import pandas as pd
from tqdm import tqdm
import numpy as np


class SplitData():
    def __init__(self, config_path):
        self.config = read_params(config_path)

        self.preprocessed_files = self.config['artifacts']['03_preprocess']['preprocessed_files']
        
        self.x_train = self.config['artifacts']['04_split_data']['x_train_dir']
        self.x_test  = self.config['artifacts']['04_split_data']['x_test_dir']
        self.y_train = self.config['artifacts']['04_split_data']['y_train_dir']
        self.y_test  = self.config['artifacts']['04_split_data']['y_test_dir']

        clean_prev_dirs_if_exists([self.x_train, self.x_test, self.y_train, self.y_test])
        create_dirs([self.x_train, self.x_test, self.y_train, self.y_test])

        self.random_state = self.config['base']['random_state']
        self.test_size = self.config['base']['test_size']

    def split_the_data(self):
        try:
            files_list = [f for f in os.listdir(self.preprocessed_files) if f[0]=='X']

            for i in tqdm(range(len(files_list))):
                
                X_data = get_data(os.path.join(self.preprocessed_files, files_list[i]))
                Y_data = get_data(os.path.join(self.preprocessed_files, 'Y' + files_list[i][1:]))
              
                x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=self.test_size, random_state=self.random_state)

                x_train.to_csv(os.path.join(self.x_train, files_list[i][2:]), index=False)
                x_test.to_csv(os.path.join(self.x_test, files_list[i][2:]), index=False)
                y_train.to_csv(os.path.join(self.y_train, files_list[i][2:]), index=False)
                y_test.to_csv(os.path.join(self.y_test, files_list[i][2:]), index=False)
            
        except Exception as e:
            raise e 


if __name__ == '__main__':
    print("\n", 10*"===", " 03 Split Data Stage ", 10*"===", "\n")
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()

    try:      
        obj = SplitData(config_path=parsed_args.config)
        obj.split_the_data()
    except Exception as e:
        raise e