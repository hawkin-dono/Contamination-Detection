# Contamination-Detection
This repository contains code and supplementary materials which are required to evaluate the contamination level of a given dataset by using various LLM: Gemma, LLama,....

## Hướng dẫn 
### Clone github repo
```
git clone https://github.com/hawkin-dono/Contamination-Detection.git
```
### Khởi tạo môi trường ảo
### Cài đặt thư viện cần thiết
```
pip install -r requirements.txt
```
### Tùy chỉnh thông số
Thay đổi device và batch_size ở đầu file eval sao cho phù hợp với phần cứng
```
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2
```
Thay đổi các thông số cần thiết để chạy evaluate
- model_names: str or list of str to search model 
- data_names: str or list of str to search data 
- size: number of sample for evaluating 
- process_types: mask_wrong_answer or mask_half_question
- prompt_prefix, prompt_sufix: str to add to prompt 
- final_prompt = prompt_prefix + precessed question core + prompt_suffix
- intermediate_data_save_path: path to save intermediate processed data
```
evaluate(model_names= None, data_names = "domain_addition", size= 10, process_types= ["mask_wrong_answer"], prompt_prefix= None, prompt_suffix=None)
```
Ví dụ: 
evaluate(model_names= ["gemma-2-9b-it", "Phi-3.5"], data_names = "domain_addition", size= 10, process_types= ["mask_wrong_answer"], prompt_prefix= None, prompt_suffix=None)
### Run project
```
python eval.py
```

