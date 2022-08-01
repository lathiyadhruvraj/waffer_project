from sklearn.model_selection import train_test_split
import argparse
import os 
from utils.common_utils import read_params, create_dirs, clean_prev_dirs_if_exists, get_data, move_files
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
from imblearn.metrics import classification_report_imbalanced
from imblearn.metrics import geometric_mean_score as gmean
from imblearn.metrics import make_index_balanced_accuracy as iba
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import json

class Evaluation():
    def __init__(self, config_path):
        self.config = read_params(config_path)

        self.model_dir = self.config['artifacts']['05_train']['train_dir']
        self.split_data_dir = self.config['artifacts']['04_split_data']['split_data_dir']

        self.results_dir = self.config['artifacts']['06_evaluate']['results_dir']
        self.best_model_dir = self.config['artifacts']['06_evaluate']['best_model_dir']
        clean_prev_dirs_if_exists([self.results_dir, self.best_model_dir])
        create_dirs([self.results_dir, self.best_model_dir])


    def __load_and_predict(self, X_test, Y_test):
        try:
            models_list = [ model for model in os.listdir(self.model_dir) if model[-4:] == ".sav" ]

            all_models_info = {}
            for i in tqdm(range(len(models_list)), "Finding best model"):
                model_path = os.path.join(self.model_dir, models_list[i])

                loaded_model = pickle.load(open(model_path, 'rb'))
                y_pred = loaded_model.predict(X_test)
         
                df = pd.DataFrame(y_pred)
                df.to_csv(os.path.join(self.results_dir, models_list[i][:-4]+".csv"), index=False)
                
                conf_matrix = confusion_matrix(Y_test, y_pred)
                imbalanced_report = classification_report_imbalanced(Y_test, y_pred, target_names=['Bad', 'Good'])
                
                # conf_mat_and_imb_rep = [str(models_list[i][:-4]), conf_matrix, imbalanced_report]
                
                roc_auc = roc_auc_score(Y_test, y_pred, average='weighted')
                f1 = f1_score(Y_test, y_pred)
                gmea = iba(alpha=0.2, squared=True)(gmean)
                gmean_score = gmea(Y_test, y_pred, average=None)
                # scores = [str(models_list[i][:]),roc_auc, f1, gmean_score]

                model_info = {  
                                "model_name" : models_list[i][:],
                                "conf_matrix" : conf_matrix,
                                "imbalanced_report" : imbalanced_report,
                                "roc_auc_score" : roc_auc,
                                "f1_score" : f1,
                                "gmean_score" : gmean_score
                            }
                
                all_models_info.update({str(models_list[i][:-4]) : model_info})

            return all_models_info

        except Exception as e:
            raise e

    def eval_models(self):
        X_test = get_data(os.path.join(self.split_data_dir, "X_test.csv"))
        Y_test = get_data(os.path.join(self.split_data_dir, "Y_test.csv")).values.ravel()
        evaluate_on =  self.config['artifacts']['06_evaluate']['evaluate_on'][0]

        all_models_info = self.__load_and_predict(X_test, Y_test)

        max_roc = 0
        model_name = ""
        for _, info in all_models_info.items():
            if max_roc < info[evaluate_on]:
                max_roc = info[evaluate_on]
                model_name = info["model_name"]
        
        model_file = os.path.join(self.model_dir, model_name)
        move_files([model_file], self.best_model_dir, remove_file=False)
        
        # with open(os.path.join(self.model_dir, "all_models_eval.json"), 'w') as f:
        #     json.dump(all_models_info, f, indent=4)


if __name__ == '__main__':
    print("\n", 10*"===", " 06 Evaluation Stage ", 10*"===", "\n")
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()

    try:      
        obj = Evaluation(config_path=parsed_args.config)
        obj.eval_models()
        print("Evaluation Completed")
    except Exception as e:
        raise e