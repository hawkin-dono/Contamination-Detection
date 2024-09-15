from HF_model import HF_Model 
# import torch 
import os 
import torch
import pandas as pd
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_model_list(path: str = "model/hf_model.csv", search_key_word: list[str]= None):
    """ 
    Inputs:
        path: csv_file contain config 
        search_key_word: str or list of string to search for model_name
    Return list of dict config 
    each config is a dict with keys: model_name, apply_chat_template, quantized, model_library"""
    df = pd.read_csv(path)
    df['apply_chat_template'] = df['apply_chat_template'].astype(bool)
    df['quantized'] = df['quantized'].astype(bool)
    if not search_key_word:
        return df.to_dict("records")
    else:
        if type(search_key_word) == str:
            search_key_word = search_key_word.lower()
            df = df[df["model_name"].str.lower().str.contains(search_key_word)]
            return df.to_dict("records")
        else:
            res = []
            for key_word in search_key_word:
                key_word = key_word.lower()
                res += df[df["model_name"].str.lower().str.contains(key_word)].to_dict("records")
            res = set(res)
            return list(res)
        

def get_data_list(path: str = "data", search_key_word: list[str] = None):
    """
    Inputs: 
        path: path of folder containing data
        search_key_word: str of list of string to search data files
    """
    files = os.listdir(path)
    if search_key_word is not None:
        if type(search_key_word) == str:
            files = [i for i in files if search_key_word in i]
        else:
            res = []
            for key_word in search_key_word:
                res += [i for i in files if key_word in i]
            res = set(res)
            files = list(res)
    return [path + "/" + file for file in files]

def extract_name(model_path: str, data_path: str):
    model_name = model_path.split("/")[-1]
    data_name = data_path.split("/")[-1][:-4]
    return model_name + "+" + data_name

def evaluate(model_names: list[str]= None, data_names: list[str]= None, 
             size: int = None, process_types= None, 
             prompt_prefix= None, prompt_suffix= None, intermediate_data_save_path= None,
             DEVICE= "cpu", BATCH_SIZE= 4):
    """
    Inputs: 
        model_names: str or list of str to search model 
        data_names: str or list of str to search data 
        size: number of sample for evaluating 
        process_types: mask_wrong_answer or mask_half_question
        prompt_prefix, prompt_sufix: str to add to prompt 
            final_prompt = prompt_prefix + precessed question core + prompt_suffix
        intermediate_data_save_path: path to save intermediate processed data
    """
    model_list = get_model_list(search_key_word= model_names)
    data_list = get_data_list(search_key_word= data_names)
    if process_types is None:
        process_types = ["mask_wrong_answer", "mask_half_question"]
    print(model_list)
    print(data_list)
    for model_config in tqdm(model_list):
        model_path = model_config["model_name"]
        model = HF_Model(model_name= model_path, device=DEVICE, apply_chat_template= model_config["apply_chat_template"], 
                         quantized= model_config["quantized"], model_library= model_config["model_library"], batch_size= BATCH_SIZE)
        for data_path in tqdm(data_list):
            for process_type in process_types:
                log_path = "logger" + "/" + extract_name(model_path, data_path) + "_" + process_type
                if os.path.exists((log_path)):
                    continue
                os.makedirs(log_path, exist_ok=True)
                res, score = model.predict_dataframe(data_path, process_type= process_type,prompt_prefix= prompt_prefix, prompt_suffix= prompt_suffix, intermediate_data_save_path= intermediate_data_save_path, size= size)
                res.to_csv(f"{log_path}/result.csv")
                with open(f"{log_path}/score.txt", "w") as f:
                    f.write(str(score))
        tqdm.write(f"{model_path} done")
    tqdm.write("All done")
    

def main():
    # prompt_prefix = """Dựa vào trí nhớ của bạn về các bộ dữ liệu, hãy điền vào đoạn <MASKED> trong câu sau để hoàn thành 1 câu hỏi trắc nghiệm. """
    # prompt_suffix = ""
    os.environ["CUDA_VISIBLE_DEVICES"]="4,5"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)
    BATCH_SIZE = 10
    evaluate(model_names= None, data_names = ["final_domain_ver1", "eng_final_domain_ver1"],  process_types= ["mask_wrong_answer", "mask_half_question"], 
             prompt_prefix= None, prompt_suffix= None,size = None,
             intermediate_data_save_path= None, DEVICE= DEVICE, BATCH_SIZE= BATCH_SIZE)
            
if __name__ == "__main__":
    main()