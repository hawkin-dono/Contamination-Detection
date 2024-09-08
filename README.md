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
Lựa chọn kích thức yêu cầu của dữ liệu bằng cách thay đổi biến size trong predict_dataframe
```
res, score = model.predict_dataframe(data_path, size= 10, process_type= "mask_half_question")
```
### Run project
```
python eval.py
```

