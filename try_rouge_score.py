# import nltk
# nltk.download('punkt')

from torchmetrics.text.rouge import ROUGEScore
from transformers import AutoTokenizer, AutoModelForCausalLM
from pprint import pprint
import re

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
preds = "Giảm thuế thu nhập từ đầu tư, cung cấp tín dụng cho đầu tư, và giảm thâm hụt"
# preds = re.sub(r"[^a-z0-9]+", " ", preds.lower())
target = " Tăng thuế thu nhập từ tiết kiệm, cung cấp tín dụng thuế đầu tư, và giảm thâm hụt"
# target = re.sub(r"[^a-z0-9]+", " ", target.lower())

# preds = "C. Cán cân trọng lượng"

def normalizer(text):
    text = re.sub(r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+", " ", text.lower())
    text = text.strip(" ")
    return text

print(tokenizer.tokenize(normalizer(preds)))
print(tokenizer.tokenize(normalizer(target)))
rouge = ROUGEScore(rouge_keys="rougeL", tokenizer=tokenizer.tokenize, use_stemmer= False, normalizer=normalizer)
# rouge = ROUGEScore(rouge_keys="rougeL")
# import inspect
# print(inspect.getsource(ROUGEScore))

print(rouge)
pprint(rouge(preds, target)["rougeL_fmeasure"])

