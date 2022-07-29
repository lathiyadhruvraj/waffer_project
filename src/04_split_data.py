from sklearn.model_selection import train_test_split
import argparse
import os 
from utils.common_utils import read_params, create_dirs, clean_prev_dirs_if_exists
from tqdm import tqdm
import pandas as pd

class SplitData():
    def __init__(self, config_path):
        self.config = read_params(config_path)

        self.preprocessed_files = self.config['artifacts']['03_preprocess']['preprocessed_files']
        
        self.split_data_dir = self.config['artifacts']['04_split_data']['split_data_dir']
        clean_prev_dirs_if_exists([self.split_data_dir])
        create_dirs([self.split_data_dir])

        self.random_state = self.config['base']['random_state'][0]
        self.test_size = self.config['base']['test_size']

    def split_the_data(self):
        try:
            X_data = pd.read_csv(os.path.join(self.preprocessed_files, "X_all.csv"))
            Y_data = pd.read_csv(os.path.join(self.preprocessed_files, "Y_all.csv"))
        
            x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=self.test_size, random_state=self.random_state)

            x_train.to_csv(os.path.join(self.split_data_dir,"X_train.csv"), index=False)
            x_test.to_csv(os.path.join(self.split_data_dir,"X_test.csv"), index=False)
            y_train.to_csv(os.path.join(self.split_data_dir,"Y_train.csv"), index=False)
            y_test.to_csv(os.path.join(self.split_data_dir,"Y_test.csv"), index=False)
                    
        except Exception as e:
            raise e 


if __name__ == '__main__':
    print("\n", 10*"===", " 04 Split Data Stage ", 10*"===", "\n")
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()

    try:      
        obj = SplitData(config_path=parsed_args.config)
        obj.split_the_data()
        print("Splitting Completed")
    except Exception as e:
        raise e