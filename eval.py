from HF_model import HF_Model 
# import torch 
import os 
import torch
import pandas as pd
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4

def get_model_list(path: str = "model/hf_path.txt"):
    
    with open(path, "r") as f:
        model_list = f.readlines()
    
    return list(map(lambda x: x.strip("\n"), model_list))

def get_data_list(path: str = "data"):
    files = os.listdir(path)
    # return [path + "/" + file for file in files]
    return ["data/masked_wrong_answer.csv"]

def extract_name(model_path: str, data_path: str):
    model_name = model_path.split("/")[-1]
    data_name = data_path.split("/")[-1][:-4]
    return model_name + "+" + data_name

def main():
    model_list = get_model_list()
    data_list = get_data_list()
    for model_path in tqdm(model_list):
        # print(model_path)
        model = HF_Model(net=model_path, device=DEVICE, batch_size=BATCH_SIZE)
        for data_path in tqdm(data_list):
            log_path = "logger" + "/" + extract_name(model_path, data_path)
            os.makedirs(log_path, exist_ok=True)
            df = pd.read_csv(data_path)
            res, score = model.predict_dataframe(df)
            res.to_csv(log_path + "/result.csv")
            with open(log_path + "/score.txt", "w") as f:
                f.write(str(score))
        tqdm.write(f"{model_path} done")
    tqdm.write("All done")
            
    
if __name__ == "__main__":
    main()