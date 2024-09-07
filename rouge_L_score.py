import os 
import pandas as pd 
from torchmetrics.text.rouge import ROUGEScore
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from tqdm import tqdm
from pathlib import Path

os.environ["HF_TOKEN"] = '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'
DIR = Path(__file__).parent.resolve()
DATA_ROOT = f'{DIR}/logger'

def normalizer(text):
    text = re.sub(r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+", " ", text.lower())
    text = text.strip(" ")
    return text


def run(path, model_name):
    data_path = f'{path}/result.csv'
    model_name = model_name
    df = pd.read_csv(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rouge = ROUGEScore(rouge_keys="rougeL", tokenizer=tokenizer.tokenize, use_stemmer= False, normalizer=normalizer)

    res = []
    for _, row in df.iterrows():
        target = str(row["Label"])
        prediction = str(row["Answer"])
        score = rouge(prediction, target)
        res.append(score)

    df["check_score"] = res
    df["check_score"] = df["check_score"].apply(lambda x: x["rougeL_fmeasure"].item())
    df.to_csv(data_path, index=False)
    
    with open(f'{path}/score.txt', 'w') as f:
        f.write(f'{df["check_score"].mean()}')

    return df["check_score"].mean()

def get_model_path(data_path): 
    model = data_path.split('+')[0]
    model_list = pd.read_csv(f'{DIR}/model/hf_model.csv')
    model_list = model_list['model_name'].tolist()
    for m in model_list:
        if model.lower() in m.lower():
            model = m
    return model

def eval_score(data):
    data_path = f'{DATA_ROOT}/{data}'
    model_path = get_model_path(data)
    run(data_path, model_path)

def main():
    all_data = os.listdir(DATA_ROOT)
    # for data in tqdm(all_data):
        # print(data)
        # eval_score(data)
    eval_score('gemma-2-2b-it+domain_masked_wrong_answer')

if __name__ == "__main__":
    main()

