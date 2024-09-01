from typing import Any, Dict, Tuple

import torch
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rouge_score import rouge_scorer 
from csv_dataset import CSV_Dataset
from tqdm import tqdm

def load_quantize(model_name):
    config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config,
        device_map="auto",
    )
    return model

class HF_Model():  
    def __init__(
        self,
        net: str, #huggingface model name
        device = "cuda",
        batch_size = 16,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(net, padding_side= 'left')
        self.model = load_quantize(net)
        # self.model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def predict_batch(self, batch):
        def stop_generation(outputs, scores):
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if generated_text.endswith("\n") and generated_text[-20:].lower().find("answer") == -1:
                return True
            return False
        x, y = batch
        with torch.no_grad():
            try:
                inputs = self.tokenizer(x, return_tensors="pt", padding= True).to(self.device)
            except:
                inputs = self.tokenizer(x, return_tensors="pt", truncation=True).to(self.device)
            generate_ids = self.model.generate(**inputs, min_new_tokens = 10, max_new_tokens= 200, pad_token_id=self.tokenizer.eos_token_id, stopping_criteria=[stop_generation])
            answer = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
            with open('answer.txt', 'a') as file:
                file.write(str(answer))
                file.write('\n')
            
        rouge = []
        for i in range(len(answer)):
            answer[i] = answer[i][len(x[i]):]
            rouge.append(self.rouge.score(answer[i], y[i])["rougeL"].fmeasure)
        
        return answer, y, rouge

    def predict_dataframe(self, ds: pd.DataFrame) -> pd.DataFrame:
        dataset = CSV_Dataset(ds)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size= self.batch_size, shuffle=False, num_workers= 4)
        
        answer, label, rouge = [], [], []
        
        for batch in tqdm(data_loader):
            tmp1, tmp2, tmp3 = (self.predict_batch(batch))
            answer.extend(tmp1)
            label.extend(tmp2)
            rouge.extend(tmp3)
        df = pd.DataFrame({"Question": ds.Question, "Answer": answer, "Label": label, "rouge_score": rouge})
        score = df["rouge_score"].mean()
        
        return df, score
    

# if __name__ == "__main__":
#     data_path = "data/mmlu_masked_wrong_answer.csv"
#     df = pd.read_csv(data_path)[:100]
    
#     model = HF_Model("Qwen/Qwen2-0.5B", "cuda:0")
#     res, score = model.predict_dataframe(df)
#     res.to_csv("result.csv")
    
#     with open("score.txt", "w") as f:
#         f.write(str(score))
    