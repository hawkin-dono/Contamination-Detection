import pandas as pd 
from torchmetrics.text.rouge import ROUGEScore
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class Rougescore():
    def __init__(self, tokenizer):
        self.rouge = ROUGEScore(rouge_keys="rougeL", tokenizer=tokenizer.tokenize, use_stemmer= False, normalizer=self.normalizer)
        
    def normalizer(self, text):
        text = re.sub(r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+", " ", text.lower())
        text = text.strip(" ")
        return text
    def compute(self, prediction, target):
        return self.rouge(prediction, target)["rougeL_fmeasure"].item()
    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    rouge_score = Rougescore(tokenizer)
    preds = "Chúng tôi đi lên nuis"
    target = " Tăng thuế thu nhập từ tiết kiệm, cung cấp tín dụng thuế đầu tư, và giảm thâm hụt"
    print(rouge_score.compute(preds, target))
    