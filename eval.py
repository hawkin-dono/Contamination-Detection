from HF_model import HF_Model 
# import torch 
import os 
import torch
import pandas as pd
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2

def get_model_list(path: str = "model/hf_model.csv", search_key_word: str = None):
    """ 
    return list of dict config 
    each config is a dict with keys: model_name, apply_chat_template, quantized, model_library"""
    df = pd.read_csv(path)
    df['apply_chat_template'] = df['apply_chat_template'].astype(bool)
    df['quantized'] = df['quantized'].astype(bool)
    if not search_key_word:
        return df.to_dict("records")
    else:
        search_key_word = search_key_word.lower()
        df = df[df["model_name"].str.lower().str.contains(search_key_word)]
        return df.to_dict("records")
        

def get_data_list(path: str = "data", search_key_word: str = None):
    files = os.listdir(path)
    if search_key_word is not None:
        files = [i for i in files if search_key_word in i]
    return [path + "/" + file for file in files]

def extract_name(model_path: str, data_path: str):
    model_name = model_path.split("/")[-1]
    data_name = data_path.split("/")[-1][:-4]
    return model_name + "+" + data_name

def main():
    # model_list = get_model_list(search_key_word= "google/gemma-2-2b-it")
    model_list = get_model_list()
    data_list = get_data_list()
    
    print(model_list)
    for model_config in tqdm(model_list):
        model_path = model_config["model_name"]
        model = HF_Model(model_name= model_path, device=DEVICE, apply_chat_template= model_config["apply_chat_template"], 
                         quantized= model_config["quantized"], model_library= model_config["model_library"], batch_size= BATCH_SIZE)
        for data_path in tqdm(data_list):
            log_path = "logger" + "/" + extract_name(model_path, data_path)
            os.makedirs(log_path, exist_ok=True)
            # df = pd.read_csv(data_path)[:10]
            df = pd.read_csv(data_path)
            res, score = model.predict_dataframe(df)
            res.to_csv(log_path + "/result.csv")
            with open(log_path + "/score.txt", "w") as f:
                f.write(str(score))
        tqdm.write(f"{model_path} done")
    tqdm.write("All done")
            
    
if __name__ == "__main__":
    main()