from typing import Any, Dict, Optional, Tuple
import pandas as pd 
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from prompt_format import Prompt_format

class CSV_Dataset(Dataset):
    def __init__(self, df: pd):
        self.data = df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.loc[idx, "Question"], self.data.loc[idx, "Label"], self.data.loc[idx, "Fed_prompt"]
    
if __name__ == "__main__":
    data_path = "data/domain.csv"
    prompt_formator = Prompt_format(data_path)
    ds: pd.DataFrame = prompt_formator.format("mask_half_question")
    print(ds.head())
    ds = ds.sample(10)
    dataset = CSV_Dataset(ds)
    print(dataset)
    print(dataset[1])
    
    