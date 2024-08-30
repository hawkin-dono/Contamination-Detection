from typing import Any, Dict, Tuple

import torch
import pandas as pd 
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
# Use a pipeline as a high-level helper
from transformers import pipeline
from rouge_score import rouge_scorer 

class HF_Model():  
    def __init__(
        self,
        net: str, #huggingface model name
    ) -> None:
        super().__init__()


        self.net = pipeline("text-generation", model=net)
        # loss function
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def forward(self, x: str) -> str:
        messages = [
            {"role": "user", "content": x},
        ]
        return self.net(messages)[0]["generated_text"]


    def predict_single_sample(
        self, batch: Tuple[str, str]
    ) -> Tuple[float, str, str]:
        x, y = batch
        answer = self.forward(x)
        rouge = self.rouge.score(answer, y)["rougeL"].fmeasure
        return answer, y, rouge

    def predict_dataframe(self, ds: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in ds.iterrows():
            results.append(self.predict_single_sample(row["Question"], row["Label"]))
        df = pd.DataFrame(
            results, columns=["rouge", "answer", "y"]
        )
        score = df["rouge"].mean()
        
        return df, score
    