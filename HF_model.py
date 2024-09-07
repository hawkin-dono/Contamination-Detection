from typing import Any, Dict, Tuple

import torch
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM, Phi3ForCausalLM
from rouge_score import rouge_scorer 
from csv_dataset import CSV_Dataset
from tqdm import tqdm
from prompt_format import Prompt_format

import os
os.environ["HF_TOKEN"] = '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'
class HF_Model():  
    def __init__(
        self,
        model_name: str, #huggingface model name
        device = "cpu",
        batch_size = 4,
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
        
    def apply_chat_template_sample(self, question: str, fed_prompt: str):
        
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": fed_prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def extract_answer(self, question, response, fed_prompt):
        if not fed_prompt:
            finding_part = question.split(".")[-2]
            index = response.find(finding_part)
            answer = response[index + len(finding_part):]
            return answer
        else:
            idx = question.find(fed_prompt)
            question = question[idx + len(fed_prompt):]
            idx = question.find(fed_prompt)
            answer = question[idx + len(fed_prompt):]
            
            return answer
            
        
    def predict_batch(self, batch):
        def stop_generation(outputs, scores):
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if generated_text.endswith("\n") and generated_text[-20:].lower().find("answer") == -1:
                return True
            return False
        x, y, fed_prompt = batch
        with torch.no_grad():
            inputs = self.tokenizer(x, return_tensors="pt", padding= True, ).to(self.device)
            # generate_ids = self.model.generate(**inputs, min_new_tokens = 10, max_new_tokens= 200, pad_token_id=self.tokenizer.eos_token_id, stopping_criteria=[stop_generation])
            generate_ids = self.model.generate(**inputs, min_new_tokens = 10, max_new_tokens= 200, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
            
        rouge = []
        answer = ["" for i in range(len(x))]
        for i in range(len(response)):
            answer[i] = self.extract_answer(x[i], response[i], fed_prompt[i])
            rouge.append(self.rouge.score(answer[i], y[i])["rougeL"].fmeasure)
        
        return answer, y, rouge

    def predict_dataframe(self, data_path: str, size: int= None, type: str = None) -> pd.DataFrame:
        """
        predict on a 
        
        Inputs:
            data_path: str: path to the data
            size: int: number of samples to predict
            type: str: type of the data: "mask_wrong_answer", "mask_half_question", "shuffle_true_answer"
        Outputs:
            df: pd.DataFrame: dataframe with 4 columns: Question, Answer, Label, rouge_score
        """
        prompt_formator = Prompt_format(data_path)
        ds: pd.DataFrame = prompt_formator.format(type)
        if size is not None:
            ds = ds.sample(size)
        if self.apply_chat_template == True: 
            for _, row in ds.iterrows():
                row["Quesiton"] = self.apply_chat_template_sample(row["Question"], row["Fed_prompt"])
        
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
        # model = HF_Model(model_name= model_name, device= device, apply_chat_template= True, quantized= True)
        model = HF_Model(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda:3", apply_chat_template=True, quantized=True,
                        model_library="LlamaForCausalLM")
        x = """Hãy điền vào trong dấu [] tại lựa chọn D dựa vào trí nhớ của bạn về các bộ dữ liệu.
    ### Câu hỏi: Trong giờ ra chơi, A trêu đùa và đánh B gây chảy máu và gãy răng, các bạn trong lớp không ai có ý kiến gì vì sợ A đánh. Trong tình huống này em sẽ làm gì?
    Lựa chọn: 
    A: Báo với cô giáo chủ nhiệm để tìm cách giải quyết.
    B: Mặc kệ vì không liên quan đến mình.
    C: []
    D: Chạy đi chỗ khác chơi.
    Hãy đưa ra câu trả lời chỉ chứa duy nhất nội dung của phần lựa chọn được yêu cầu.
    """
        x = model.apply_chat_template_sample(x)
        print(x)
        y = "Chạy đi chỗ khác chơi."
        answer, y, rouge = model.predict_batch(([x], [y]))
        print(answer)
        print(rouge)
    def test_dataframe():
        data_path = "data/domain.csv"
        
        model = HF_Model(model_name="Qwen/Qwen2-0.5B", device="cuda:0", apply_chat_template=True, quantized=True,
                        model_library="AutoModelForCausalLM")
        res, score = model.predict_dataframe(data_path, type= "mask_half_question", size= 10)
        res.to_csv("result.csv")
        
        with open("score.txt", "w") as f:
            f.write(str(score))
        
    # test_single_sample("microsoft/Phi-3.5-mini-instruct", "cuda:0")
    test_dataframe()        