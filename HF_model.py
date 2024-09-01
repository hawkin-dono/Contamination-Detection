from typing import Any, Dict, Tuple

import torch
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM, Phi3ForCausalLM
from rouge_score import rouge_scorer 
from csv_dataset import CSV_Dataset
from tqdm import tqdm

import os
os.environ["HF_TOKEN"] = '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'
class HF_Model():  
    def __init__(
        self,
        model_name: str, #huggingface model name
        device = "cpu",
        batch_size = 16,
        quantized: bool = False,
        apply_chat_template = False ,
        model_library: str = "AutoModelForCausalLM"  
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side= 'left')
        self.model = self.load_model(model_name, model_library, quantized)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        self.device = device
        self.batch_size = batch_size
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.apply_chat_template = apply_chat_template
        
    def load_model(self, model_name: str, model_library: str, quantized: bool = False):
        access_token = "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"
        # access_token = "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"
        config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
        if model_library == "AutoModelForCausalLM":
            if quantized == True:
                model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=config,
                        device_map="auto",
                        token = access_token
                    )
            else: 
                model = AutoModelForCausalLM.from_pretrained(model_name, token= access_token)
        elif model_library == "LlamaForCausalLM":
            if quantized: 
                model = LlamaForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=config,
                        device_map="auto",
                        token= access_token
                    )
            else: 
                model = LlamaForCausalLM.from_pretrained(model_name, token= access_token)
        elif model_library == "Phi3ForCausalLM":
            if quantized: 
                model = Phi3ForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=config,
                        device_map="auto",
                        token= access_token
                    )
            else: 
                model = Phi3ForCausalLM.from_pretrained(model_name, token= access_token)
        else:
            raise ValueError("model_library must be one of the following: AutoModelForCausalLM, LlamaForCausalLM, Phi3ForCausalLM")
        return model
        
    def apply_chat_template_sample(self, s: str):
        
        messages = [{"role": "user", "content": s}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def predict_batch(self, batch):
        def stop_generation(outputs, scores):
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if generated_text.endswith("\n") and generated_text[-20:].lower().find("answer") == -1:
                return True
            return False
        x, y = batch
        # print(x)
        with torch.no_grad():
            inputs = self.tokenizer(x, return_tensors="pt", padding= True, ).to(self.device)
            # generate_ids = self.model.generate(**inputs, min_new_tokens = 10, max_new_tokens= 200, pad_token_id=self.tokenizer.eos_token_id, stopping_criteria=[stop_generation])
            generate_ids = self.model.generate(**inputs, min_new_tokens = 10, max_new_tokens= 200, pad_token_id=self.tokenizer.eos_token_id)
            answer = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
            
        rouge = []
        # print(answer)
        for i in range(len(answer)):
            finding_part = x[i].split(".")[-2]
            index = answer[i].find(finding_part)
            answer[i] = answer[i][index + len(finding_part):]
            rouge.append(self.rouge.score(answer[i], y[i])["rougeL"].fmeasure)
        # print(answer)
        
        return answer, y, rouge

    def predict_dataframe(self, ds: pd.DataFrame) -> pd.DataFrame:
        if self.apply_chat_template == True: 
            ds.Question = ds.Question.apply(self.apply_chat_template_sample)
        dataset = CSV_Dataset(ds)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size= self.batch_size, shuffle=False, num_workers= 4)
        
        answer, label, rouge = [], [], []
        
        for batch in tqdm(data_loader):
            tmp1, tmp2, tmp3 = self.predict_batch(batch)
            answer.extend(tmp1)
            label.extend(tmp2)
            rouge.extend(tmp3)
        df = pd.DataFrame({"Question": ds.Question, "Answer": answer, "Label": label, "rouge_score": rouge})
        score = df["rouge_score"].mean()
        
        return df, score
    

if __name__ == "__main__":
    def test_single_sample(model_name: str, device: str):
        model = HF_Model(model_name= model_name, device= device, apply_chat_template= True, quantized= True)
        x = """Hãy điền vào trong dấu [] tại lựa chọn D dựa vào trí nhớ của bạn về các bộ dữ liệu.
Hãy nhớ một luật quan trọng rằng: câu trả lời của bạn phải khác các lựa chọn còn lại
Câu hỏi: Trong giờ ra chơi, A trêu đùa và đánh B gây chảy máu và gãy răng, các bạn trong lớp không ai có ý kiến gì vì sợ A đánh. Trong tình huống này em sẽ làm gì?
Lựa chọn: 
A: Báo với cô giáo chủ nhiệm để tìm cách giải quyết.
B: Mặc kệ vì không liên quan đến mình.
C: Cùng với A đánh B cho vui.
D: []
Hãy đưa ra câu trả lời chỉ chứa duy nhất nội dung của phần lựa chọn được yêu cầu.
"""
        x = model.apply_chat_template_sample(x)
        y = "Chạy đi chỗ khác chơi."
        answer, y, rouge = model.predict_batch(([x], [y]))
        print(answer)
        print(rouge)
    def test_dataframe():
        data_path = "data/domain_masked_wrong_answer.csv"
        df = pd.read_csv(data_path)[:10]
        
        model = HF_Model(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda:0", apply_chat_template=True, quantized=True,
                        model_library="LlamaForCausalLM")
        res, score = model.predict_dataframe(df)
        res.to_csv("result.csv")
        
        with open("score.txt", "w") as f:
            f.write(str(score))
        
    test_single_sample("microsoft/Phi-3.5-mini-instruct", "cuda:0")
    # test_dataframe()        