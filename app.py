import streamlit as st
import pandas as pd
import os
import argparse
import shutil
from predict.check_files import main
import yaml
import pickle


args = argparse.ArgumentParser()
args.add_argument('--config', default='predict.yaml')
args.add_argument('--schema', default='schema.yaml')
parsed_args = args.parse_args()
config_path = parsed_args.config
with open(config_path) as yaml_file:
    config = yaml.safe_load(yaml_file)


save_file_dir = config["predict"]["valid_and_preprocess"]["valid_and_preprocess_dir"]
save_path = os.path.join(os.getcwd(), save_file_dir)
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path, exist_ok=True)

model_dir = config["predict"]["model_dir"]
model_name = [ model for model in os.listdir(model_dir) if model[-4:] == ".sav" ]
model_path = os.path.join(model_dir, model_name[0])

loaded_model = pickle.load(open(model_path, 'rb'))
pred_file_dir = config['predict']['valid_and_preprocess']['preprocessed_files']


st.title("Wafer Prediction Project")

choice = st.radio ("Choose From Below Options: ", ["Upload Own File Here", "Show Available Files"])

try:
    if choice == "Show Available Files":
    
        files_dir = os.listdir(config['predict']['files_for_pred'])
        
        chosen_file = st.radio ("Choose From Below Options: ", files_dir)

        st.write("PREDICTION OF ", chosen_file)

        pred_file_path = os.path.join(config['predict']['files_for_pred'], chosen_file)
        shutil.copy(pred_file_path, save_file_dir)

        status, e = main()
        if status:
            st.success(e)

            pred_preprocessed = os.path.join(pred_file_dir, "predict_file.csv")
            pred_file = pd.read_csv(pred_preprocessed)
            result = loaded_model.predict(pred_file)

            df = pd.DataFrame(result)
            df = df.replace(0, "Bad Wafer")
            df = df.replace(1, "Good Wafer")

            st.write(df)
        
        else:
            # st.error(e)
            st.error("FILE VALIDATION FAILED - CHANGE FILE")
            st.warning("Only 591 columns allowed")
        
    
    if choice == "Upload Own File Here":
        st.subheader("Drag and Drop your File ")
        csv_file = st.file_uploader("Upload csv file", type=["csv"])

        if csv_file is not None:

            # To See details
            file_details = {"filename":csv_file.name, "filetype":csv_file.type,
                            "filesize":csv_file.size}
            st.write(file_details)
            
            if st.button("Process"):
                df = pd.read_csv(csv_file)
                st.dataframe(df)
                df.to_csv(os.path.join(save_file_dir, csv_file.name), index=None)

                status, e = main()
                if status :
                    st.success(e)
                    
                    pred_file_path = os.path.join(pred_file_dir, "predict_file.csv")
                    pred_file = pd.read_csv(pred_file_path)
                    result = loaded_model.predict(pred_file)
            
                    df = pd.DataFrame(result)
                    df = df.replace(0, "Bad Wafer")
                    df = df.replace(1, "Good Wafer")
                    st.write(df)

                else:
                    # st.error(e)
                    st.error("FILE VALIDATION FAILED - CHANGE FILE")
                    st.warning("Only 591 columns allowed")
    
except Exception as e:
    raise e
            

