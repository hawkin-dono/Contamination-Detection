from pathlib import Path

import numpy as np
import pandas as pd
from torch import tensor

import matplotlib.pyplot as plt

import re

DIRECTORY = Path(__file__).resolve().parent
DATA_PATH = DIRECTORY / 'logging_file'

def merge_csv():
    data_list = list(DATA_PATH.iterdir())
    df_list = []
    for file in data_list:
        df = pd.read_csv(file)
        df['model'] = file.stem
        if df['score'].dtype != 'float64':
            df['score'] = df['score'].apply(lambda x: eval(x)).astype('float64')
        df_list.append(df)
    df_total = pd.concat(df_list)
    # df_total['score'] = df_total['score'].apply(lambda x: eval(x)).astype('float64')
    df_total.to_csv(DATA_PATH / 'data.csv', index=False)


def cal_mean(file):
    df = pd.read_csv(file)
    if df['score'].dtype != 'float64':
        df['score'] = df['score'].apply(lambda x: eval(x)).astype('float64')
        df.to_csv(file, index=False)
    return df['score'].mean()


def pre_filter_mean():
    open(DIRECTORY / 'pre_filter_mean.txt', 'w').close() # clear the file
    data_list = list(DATA_PATH.iterdir())
    for file in data_list:
        with open(DIRECTORY / 'pre_filter_mean.txt', 'a') as f:
            f.write(f'{file.stem}: {cal_mean(file)}\n')


def plot(file):
    name = file.stem

    df = pd.read_csv(file)

    plt.figure(figsize=(10, 6))
    plt.xlabel('score')
    plt.ylabel('count')
    plt.title(f'Score distribution of {name}') 
    plt.hist(df['score'], bins=20)
    
    plt.savefig(DIRECTORY / 'plots' / f'{name}.png')
    plt.close()


def pre_filter_plot():
    data_list = list(DATA_PATH.iterdir())
    for file in data_list:
        plot(file)


def check_number_latex(string):
    text = re.search(r"[0-9$]+", string)
    return text is not None


def grid_search(no_math = False):
    data_list = list(DATA_PATH.iterdir())
    data_list = [x.name for x in data_list if x.stem != 'data']
    thresholds = np.arange(0, 1, 0.05)
    
    for data in data_list:
        if no_math:
            log_file = DIRECTORY / f'filter_logs_no_math/{Path(data).stem}_filtered.txt'
        else:
            log_file = DIRECTORY / f'filter_logs/{Path(data).stem}.txt'
        open(log_file, 'w').close()
        for threshold in thresholds:
            df = pd.read_csv(DATA_PATH / data)
            if no_math:
                df['num'] = df['predict'].astype('str').apply(check_number_latex)
                cond = (df['score'] > threshold) & (df['num'] == False)
                retained = df.drop(df[cond].index)
                filtered = retained[retained['num'] == False]
                with open(log_file, 'a') as f:
                    f.write(f'Threshold:{threshold},  retain_rate:{len(retained)/len(df)}, average:{filtered['score'].mean()}\n')
            else:
                cond = (df['score'] <= threshold)
                filtered = df[cond]
                with open(log_file, 'a') as f:
                    f.write(f'Threshold:{threshold}, retain_rate:{len(filtered)/len(df)}, average:{filtered['score'].mean()}\n')
    
    return 

def recal_score(model, threshold):
    filter_df = pd.read_csv(DATA_PATH / model)
    # filter_df['num'] = filter_df['predict'].astype('str').apply(check_number_latex)
    cond = (filter_df['score'] > threshold)
    filter_df['retained'] = True
    filter_df.loc[cond, 'retained'] = False
    filter_df = filter_df['retained']

    target = DIRECTORY / f'{model}_filter.txt'
    open(target, 'w').close()
    data_list = list(DATA_PATH.iterdir())
    for file in data_list:
        if file.stem == 'data':
            continue
        df = pd.read_csv(file)
        subject_df = pd.read_csv(DIRECTORY / 'domain_masked_wrong_answer.csv')['Subject']
        df = pd.concat([df, subject_df, filter_df], axis=1)
        df.to_csv(DIRECTORY / 'filtered_data' / file.name, index=False)
        cond = (df['retained'] == True)
        df = df[cond]
        with open(target, 'a') as f:
            f.write(f'{file.stem}: {df["score"].mean()}\n')
    plot_dropped(DIRECTORY / 'filtered_data' / f'{model}')

def plot_dropped(file):
    name = file.stem
    df = pd.read_csv(file)
    fig = plt.figure(figsize = (16, 9))
    count = {}
    count_drop = {}
    for key in df['Subject'].unique():
        count[key] = [df[df['Subject'] == key].shape[0]]
        count_drop[key] = [df[(df['Subject'] == key) & (df['retained'] == False)].shape[0]]

    count = pd.DataFrame.from_dict(count, orient='index')
    count_drop = pd.DataFrame.from_dict(count_drop, orient='index')
    plt.bar(count.index, count[0], color = 'b')
    plt.bar(count_drop.index, count_drop[0], color = 'r')
    plt.title(f'{name}')
    plt.savefig(DIRECTORY / 'plots_drop' / f'{name}.png')
    plt.close()

if __name__ == '__main__':
    # merge_csv()
    # pre_filter_mean()
    # pre_filter_plot()
    # grid_search()
    # grid_search(True)
    recal_score('vistral_7b_mini_logging_file_domain.csv', 0.4)
    pass