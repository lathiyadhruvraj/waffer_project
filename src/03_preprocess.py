import argparse
import os 
from utils.common_utils import read_params, create_dirs, clean_prev_dirs_if_exists
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.impute import KNNImputer
 
class Preprocess():

    def __init__(self, config_path):
        self.config = read_params(config_path)

        self.artifacts_good_files = self.config['artifacts']['02_validation']['good_files_dir']
        
        self.preprocess_dir = self.config['artifacts']['03_preprocess']['preprocess_dir']
        self.preprocessed_files_dir = self.config['artifacts']['03_preprocess']['preprocessed_files']
        
        clean_prev_dirs_if_exists([self.preprocess_dir, self.preprocessed_files_dir])
        create_dirs([self.preprocess_dir, self.preprocessed_files_dir])


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

    def __is_null_present(self, X, fname):
        null_present = False
        try:
            null_counts=X.isna().sum() # check for the count of null values per column
            for i in null_counts:
                if i > 0:
                    null_present = True
                    break
            if null_present: 
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = X.columns
                dataframe_with_null['missing values count'] = np.asarray(X.isna().sum())
                dataframe_with_null.to_csv(os.path.join(self.preprocess_dir, fname))  # storing the null column information to file
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

    def __get_columns_with_zero_std_deviation(self, data):
        columns=data.columns
        data_n = data.describe()
        col_to_drop=[]
        try:
            for x in columns:
                if (data_n[x]['std'] < 0.001) and (data_n[x]['std'] > -0.001): # check if standard deviation is zero
                # if data_n[x]['std'] == 0:
                    col_to_drop.append(x)  # prepare the list of columns with standard deviation zero
            return col_to_drop
        except Exception as e:
            raise e
    
    def __high_correlation_drop(self, data):
        try:	    
            # Create correlation matrix
            corr_matrix = data.corr().abs()

            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            # Find features with correlation greater than 0.95
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

            # Drop features 
            data.drop(to_drop, axis=1, inplace=True)

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
            
            columns = self.config['base']['remove_cols']
            useful_data = self.__remove_columns(all_files_data, columns)

            target_column_name = self.config['base']['target_col']
            X, Y = self.__separate_label_feature(useful_data, target_column_name)

            fname = self.config['artifacts']['03_preprocess']['null_cols_fname']
            null_present = self.__is_null_present(X, fname)

            if null_present:
                n_neighbors = self.config['hyperparams']['KNNImputer']['n_neighbors']
                weights = self.config['hyperparams']['KNNImputer']['weights']
                X = self.__impute_missing_values(X, n_neighbors, weights)

            col_to_drop = self.__get_columns_with_zero_std_deviation(X)
            X = self.__remove_columns(X, col_to_drop)

            self.__high_correlation_drop(X)
            
            Y = Y.replace(-1, 0)
            X.to_csv(os.path.join(self.preprocessed_files_dir, "X_all.csv"), index=False, header=False)
            Y.to_csv(os.path.join(self.preprocessed_files_dir, 'Y_all.csv'), index=False, header=False)

        except Exception as e:
            raise e



if __name__ == '__main__':
    print("\n", 10*"===", " 03 Preprocess Stage ", 10*"===", "\n")
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()

    try:      
        obj = Preprocess(config_path=parsed_args.config)
        obj.preprocessing_controller()
    except Exception as e:
        raise e