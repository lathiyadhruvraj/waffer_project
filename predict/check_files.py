import os 
import yaml
import shutil 
import re
import pandas as pd
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.impute import KNNImputer


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
        
class File_Validation():

    def __init__(self,config_path, schema_path):
        self.config = read_params(config_path)
        self.schema = read_params(schema_path)
        
        self.regex = "['wafer']+['\_'']+[\d_]+[\d]+\.csv"
        
        self.artifacts_good_files = self.config['predict']['valid_and_preprocess']['good_files_dir']
        self.artifacts_bad_files = self.config['predict']['valid_and_preprocess']['bad_files_dir']
        self.valid_and_preprocess_dir = self.config['predict']['valid_and_preprocess']['valid_and_preprocess_dir']
    

    #======== Length Of Date Stamp / Length Of Time Stamp ===================# 

    def __len_of_date_time_stamp_check(self, waffer_files_name):
        try:
            for i in tqdm(range(len(waffer_files_name)), desc="date_time_stamp_check      => "):
                filename = waffer_files_name[i]

                file = os.path.join(self.valid_and_preprocess_dir, filename)
                if (re.match(self.regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = (re.split('_', splitAtDot[0]))
                    if len(splitAtDot[1]) == self.schema["LengthOfDateStampInFile"]:
                        if len(splitAtDot[2]) == self.schema["LengthOfTimeStampInFile"]:
                            move_files([file], self.artifacts_good_files)
                        else:
                            move_files([file], self.bad_files)
                    else:
                        move_files([file], self.bad_files)
                else:
                    move_files([file], self.bad_files)

            good_files_name_1 = os.listdir(self.artifacts_good_files)
            return good_files_name_1
        
        except Exception as e:
                raise e

    #======================== Col length Check ==============================#

    def __col_length_check(self, good_files_name_1):
        try:
            for i in tqdm(range(len(good_files_name_1)), desc="col_length_check           => "):
                filename = good_files_name_1[i]
                file = os.path.join(self.good_files, filename)
                csv = pd.read_csv(file)
                
                if csv.shape[1] == (self.schema["NumberofColumns"] - 1):
                    pass
                else:
                    move_files([file], self.bad_files, remove_file=True)
            
            good_files_name_2 = os.listdir(self.artifacts_good_files)
            return good_files_name_2
        
        except Exception as e:
                raise e

    #======================== Missing Vals in Cols Check ============+++=====#

    def __missing_vals_in_cols_check(self, good_files_name_2):
        try:
            for i in tqdm(range(len(good_files_name_2)), desc="missing_vals_in_cols_check => "):
                filename = good_files_name_2[i]

                file = os.path.join(self.good_files, filename)
                csv = pd.read_csv(file)
                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count+=1

                        move_files([file], self.bad_files, remove_file=True)
                        break
                if count==0:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    csv.to_csv(file, index=None, header=True)
            
            good_files_name_3 = os.listdir(self.artifacts_good_files)
            return good_files_name_3
        
        except Exception as e:
                raise e
    
    #======================== fillna with Null ===============================#

    def __fillna_with_NULL(self, only_good_files):
        try:
            for i in tqdm(range(len(only_good_files)), desc="fillna_with_NULL           => "):
                filename = only_good_files[i]

                file = os.path.join(self.good_files, filename)
                csv = pd.read_csv(file)
                csv.fillna('NULL',inplace=True)
                # csv['Wafer'] = csv['Wafer'].str[6:]
                csv.to_csv(file, index=None, header=True)

        except Exception as e:
            raise e

    #======================== validation control def ===============================#

    def files_check_controller(self):
        try:
            self.good_files = os.path.join(os.getcwd(), self.artifacts_good_files)
            self.bad_files = os.path.join(os.getcwd(), self.artifacts_bad_files)

            clean_prev_dirs_if_exists([self.good_files, self.bad_files])
            create_dirs([self.good_files, self.bad_files])

            waffer_files_name = [f for f in os.listdir(self.valid_and_preprocess_dir)]
            
            good_files_name_1 = self.__len_of_date_time_stamp_check(waffer_files_name)
            good_files_name_2 = self.__col_length_check(good_files_name_1)

            only_good_files = self.__missing_vals_in_cols_check(good_files_name_2)

            self.__fillna_with_NULL(only_good_files)
            print("\n only good files:-", only_good_files)
        
        except Exception as e:
            print(e)
            raise e

#-----------------------------------------------------------------------------------------------------
#/////////////////////////////////////////////////////////////////////////////////////////////////////

class Preprocess():

    def __init__(self, config_path):
        self.config = read_params(config_path)

        self.artifacts_good_files = self.config['predict']['valid_and_preprocess']['good_files_dir']
        
        # self.preprocess_dir = self.config['predict']['valid_and_preprocess']['preprocess_dir']
        self.preprocessed_files_dir = self.config['predict']['valid_and_preprocess']['preprocessed_files']
        
        clean_prev_dirs_if_exists([self.preprocessed_files_dir])
        create_dirs([ self.preprocessed_files_dir])


    #============================= Get Data from 02 Stage =============================#

    def __get_data(self, file_path):
        try:
            data = pd.read_csv(file_path) # reading the data file
            return data
        except Exception as e:
            raise e

    #============================= Remove Columns =====================================#
    def __remove_columns(self, data, columns):

        try:
            final_data = data.drop(labels=columns, axis=1) # drop the labels specified in the columns
            return final_data
        except Exception as e:
            raise e

    #============================= Separate X, Y ======================================#

    def __separate_label_feature(self, useful_data, target_column_name):
        try:
            X = useful_data.drop(labels=target_column_name,axis=1)
            Y = useful_data[target_column_name] # Filter the Label columns
            return X, Y
        except Exception as e:
            raise e

   #============================= Count Null Values ===================================#

    def __is_null_present(self, X):
        null_present = False
        try:
            null_counts=X.isna().sum() # check for the count of null values per column
            for i in null_counts:
                if i > 0:
                    null_present = True
                    break
            # if null_present: 
            #     dataframe_with_null = pd.DataFrame()
            #     dataframe_with_null['columns'] = X.columns
            #     dataframe_with_null['missing values count'] = np.asarray(X.isna().sum())
            #     dataframe_with_null.to_csv(os.path.join(self.preprocess_dir, fname))  # storing the null column information to file
            return null_present
        except Exception as e:
            raise e

   #============================= KNN Imputation ======================================#

    def __impute_missing_values(self, data, n_neighbors, weights):
        try:
            imputer=KNNImputer(n_neighbors=n_neighbors, weights=weights, missing_values=np.nan)
            new_array=imputer.fit_transform(data) # impute the missing values
            # convert the nd-array returned in the step above to a Dataframe
            new_data=pd.DataFrame(data=new_array, columns=data.columns)
            return new_data
        except Exception as e:
            raise e

   #============================= Remove 0 std cols ===================================#

    def __get_columns_with_zero_std_deviation(self):
        try:
            df = pd.read_csv(self.config['predict']['valid_and_preprocess']['drop_0std_fname'])
            
            columns = []
            for i in range(df.shape[0]):
                columns.append(df.values[i][0])
            
            return columns
            
        except Exception as e:
            raise e
    
    def __high_correlation_drop(self):
        try:	    
            df = pd.read_csv(self.config['predict']['valid_and_preprocess']['cols_to_drop'])
            
            columns = []
            for i in range(df.shape[0]):
                columns.append(df.values[i][0])
            
            return columns
        except Exception as e:
            raise e

   #============================= Preprocess def controller +=========================#

    def preprocessing_controller(self):
        try:
            files_list = os.listdir(self.artifacts_good_files)
            all_files_data = pd.DataFrame()

            for i in tqdm(range(len(files_list))):
                filename = files_list[i]

                data = self.__get_data(os.path.join(self.artifacts_good_files, filename))

                all_files_data = pd.concat([all_files_data, data])
            
            columns = self.config['predict']['valid_and_preprocess']['remove_cols']
            useful_data = self.__remove_columns(all_files_data, columns)

            # target_column_name = self.config['predict']['valid_and_preprocess']['target_col']
            # X, Y = self.__separate_label_feature(all_files_data, target_column_name)

            # fname = self.config['predict']['valid_and_preprocess']['null_cols_fname']
            null_present = self.__is_null_present(useful_data)

            if null_present:
                n_neighbors = self.config['hyperparams']['KNNImputer']['n_neighbors']
                weights = self.config['hyperparams']['KNNImputer']['weights']
                X = self.__impute_missing_values(useful_data, n_neighbors, weights)

            col_to_drop = self.__get_columns_with_zero_std_deviation()
            X = self.__remove_columns(X, col_to_drop)

            cols = self.__high_correlation_drop()
            X = self.__remove_columns(X, cols)
            
            # Y = Y.replace(-1, 0)
            X.to_csv(os.path.join(self.preprocessed_files_dir, "predict_file.csv"), index=False, header=False)
            # Y.to_csv(os.path.join(self.preprocessed_files_dir, 'Y_all.csv'), index=False, header=False)

        except Exception as e:
            print(e)
            raise e

def main():
    print("\n", 10*"===", " PREDICT -> VALIDATING and PREPROCESSING FILES ", 10*"===", "\n")
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='predict.yaml')
    args.add_argument('--schema', default='schema.yaml')
    parsed_args = args.parse_args()

    try:      
        obj = File_Validation(config_path=parsed_args.config, schema_path=parsed_args.schema)
        obj.files_check_controller()

        obj2 = Preprocess(config_path=parsed_args.config)
        obj2.preprocessing_controller()
        return 1, "file is preprocessed, wait for prediction"
    except Exception as e:
        print(e)
        return 0, e