import argparse
import os 
from utils.common_utils import read_params, create_dirs, clean_prev_dirs_if_exists, move_files
import re
import pandas as pd
from tqdm import tqdm

class File_Validation():

    def __init__(self,config_path, schema_path):
        self.config = read_params(config_path)
        self.schema = read_params(schema_path)
        
        self.regex = "['wafer']+['\_'']+[\d_]+[\d]+\.csv"
        
        self.artifacts_good_files = self.config['artifacts']['02_validation']['good_files_dir']
        self.artifacts_bad_files = self.config['artifacts']['02_validation']['bad_files_dir']
        self.waffer_files_dir = self.config['artifacts']['02_validation']['s3_data_dir']
    

    #======== Length Of Date Stamp / Length Of Time Stamp ===================# 

    def __len_of_date_time_stamp_check(self, waffer_files_name):
        try:
            for i in tqdm(range(len(waffer_files_name)), desc="date_time_stamp_check      => "):
                filename = waffer_files_name[i]

                file = os.path.join(self.waffer_files_dir, filename)
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
                if csv.shape[1] == self.schema["NumberofColumns"]:
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
                # #csv.update("'"+ csv['Wafer'] +"'")
                # csv.update(csv['Wafer'].astype(str))
                csv['Wafer'] = csv['Wafer'].str[6:]
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

            waffer_files_name = [f for f in os.listdir(self.waffer_files_dir)]
            
            good_files_name_1 = self.__len_of_date_time_stamp_check(waffer_files_name)
            good_files_name_2 = self.__col_length_check(good_files_name_1)
            only_good_files = self.__missing_vals_in_cols_check(good_files_name_2)
            self.__fillna_with_NULL(only_good_files)
            print("\n only good files:-", only_good_files)
        
        except Exception as e:
            raise e


if __name__ == '__main__':
    print("\n", 10*"===", " 02 Validation Stage ", 10*"===", "\n")
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    args.add_argument('--schema', default='schema.yaml')
    parsed_args = args.parse_args()

    try:      
        obj = File_Validation(config_path=parsed_args.config, schema_path=parsed_args.schema)
        obj.files_check_controller()
    except Exception as e:
        raise e