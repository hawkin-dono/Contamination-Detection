from typing import Any, Dict, Optional, Tuple
import pandas as pd 
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

class CSV_Dataset(Dataset):
    def __init__(self, df: pd):
        self.data = df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.loc[idx, "Question"], self.data.loc[idx, "Label"]