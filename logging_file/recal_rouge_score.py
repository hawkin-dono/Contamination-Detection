from pathlib import Path

import pandas as pd
import numpy as np

DIRECTORY = Path(r'C:\Users\Admin\Desktop\Contamination-Detection\logging_file')
DATA_PATH = DIRECTORY / r'all_filter_data_log\hq_csv'

import re
import string

def normalize_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    
    return text

def lcs(pred, ref):
    len_pred = len(pred)
    len_ref = len(ref)
    
    dp = [[0] * (len_ref + 1) for _ in range(len_pred + 1)]
    
    for i in range(1, len_pred + 1):
        for j in range(1, len_ref + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[len_pred][len_ref]

def rouge_l_score(pred, ref):
    # normalize
    pred = normalize_text(pred)
    ref = normalize_text(ref)
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    
    # get lcs
    lcs_length = lcs(pred_tokens, ref_tokens)
    
    # precision & recall
    precision = lcs_length / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    f1_score = ((2 * precision * recall) / (precision + recall)) if (precision + recall != 0) else 0.0
    
    return f1_score


def main():
    file_list = list(DATA_PATH.glob('*.csv'))
    for file in file_list:
        df = pd.read_csv(file)
        df.rename(columns={'Answer': 'predict', 'Label': 'label'}, inplace=True)
        df['label'] = df['label'].astype(str)
        df['score'] = df.apply(lambda row: rouge_l_score(str(row['predict']), str(row['label'])), axis=1)
        df = df.drop(columns=df.columns[df.columns.str.contains('Unnamed')])
        df.to_csv(file, index=False)


if __name__ == '__main__':
    main()