from pathlib import Path

import numpy as np
import pandas as pd
from torch import tensor

import matplotlib.pyplot as plt

import re

DIRECTORY = Path(__file__).resolve().parent
# DATA_PATH = DIRECTORY / 'final_logger_data' / 'original'
DATA_PATH = DIRECTORY / r'all_filter_data_log/hq_csv'

def merge_csv():
    data_list = list(DATA_PATH.glob('*.csv'))
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
    open(DATA_PATH / 'pre_filter_mean.txt', 'w').close() # clear the file
    data_list = list(DATA_PATH.glob('*.csv'))
    for file in data_list:
        with open(DATA_PATH / 'pre_filter_mean.txt', 'a') as f:
            f.write(f'{file.stem}: {cal_mean(file)}\n')


def plot(file):
    name = file.stem

    df = pd.read_csv(file)

    plt.figure(figsize=(10, 6))
    plt.xlabel('score')
    plt.ylabel('count')
    plt.title(f'Score distribution of {name}') 
    plt.hist(df['score'], bins=20)
    
    TARGET = DATA_PATH / 'plots'
    TARGET.mkdir(exist_ok=True, parents=True)
    plt.savefig(TARGET / f'{name}.png')
    plt.close()


def pre_filter_plot():
    data_list = list(DATA_PATH.glob('*.csv'))
    for file in data_list:
        plot(file)


def check_number_latex(string):
    text = re.search(r"[0-9$+\-*/=]+", string)
    return text is not None


def grid_search(no_math = False):
    data_list = list(DATA_PATH.glob('*.csv'))
    data_list = [x.name for x in data_list if x.stem != 'data']
    thresholds = np.arange(0, 1, 0.05)
    
    for data in data_list:
        if no_math:
            TARGET = DATA_PATH / 'filter_logs_no_math'
            TARGET.mkdir(exist_ok=True, parents=True)
            log_file = TARGET / f'{Path(data).stem}_filtered.txt'
        else:
            TARGET = DATA_PATH / 'filter_logs'
            TARGET.mkdir(exist_ok=True, parents=True)
            log_file = TARGET / f'{Path(data).stem}.txt'
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

def recal_score(model, threshold, drop_math = False):
    # Mark which samples are to be retained
    filter_df = pd.read_csv(DATA_PATH / model)
    # filter_df['num'] = filter_df['predict'].astype('str').apply(check_number_latex)
    cond = (filter_df['score'] > threshold)
    filter_df['retained'] = True
    filter_df.loc[cond, 'retained'] = False
    filter_df = filter_df['retained']

    # Target file
    target = DATA_PATH / f'{model}_filter.txt'
    
    open(target, 'w').close() # clear the file

    # Go through all the data files
    data_list = list(DATA_PATH.glob('*.csv'))
    for file in data_list:
        if file.stem == 'data':
            continue
        # Preprocessing
        df = pd.read_csv(file)
        df = pd.concat([df, filter_df], axis=1)

        # Dropping
        TARGET = DATA_PATH / 'filtered_data'
        TARGET.mkdir(exist_ok=True, parents=True)
        df.to_csv(TARGET / file.name, index=False)
        
        # Calculate the mean
        cond = (df['retained'] == True)
        df = df[cond]
        with open(target, 'a') as f:
            f.write(f'{file.stem}: {df["score"].mean()}\n')
        
    # Plotting
    plot_dropped(DATA_PATH / 'filtered_data' / f'{model}')

def filter(model, threshold):
    # Mark which samples are to be retained
    filter_df = pd.read_csv(DATA_PATH / model)
    # filter_df['num'] = filter_df['predict'].astype('str').apply(check_number_latex)
    cond = (filter_df['score'] > threshold) | (filter_df['predict'] == filter_df['label'])
    filter_df['retained'] = True
    filter_df.loc[cond, 'retained'] = False

    TARGET = DATA_PATH / 'filtered_data'
    TARGET.mkdir(exist_ok=True, parents=True)
    filter_df.to_csv(TARGET / model, index=False)
    
    # Plotting
    # plot_dropped(DATA_PATH / 'filtered_data' / f'{model}')


def plot_dropped(file, drop_math = False):
    name = file.stem
    df = pd.read_csv(file)
    fig = plt.figure(figsize = (16, 9))
    count = {} 
    count_drop = {}
    for key in df['Grade'].unique():
        count[key] = [df[df['Grade'] == key].shape[0]]
        count_drop[key] = [df[(df['Grade'] == key) & (df['retained'] == False)].shape[0]]

    count = pd.DataFrame.from_dict(count, orient='index')
    count_drop = pd.DataFrame.from_dict(count_drop, orient='index')
    plt.bar(count.index, count[0], color = 'b')
    plt.bar(count_drop.index, count_drop[0], color = 'r')
    plt.title(f'{name}')
    TARGET = DATA_PATH / 'plots_drop'
    TARGET.mkdir(exist_ok=True, parents=True)
    if drop_math:
        plt.savefig(TARGET/ f'{name}_no_math.png')
    else:
        plt.savefig(TARGET / f'{name}.png')
    plt.close()


if __name__ == '__main__':
    # merge_csv()
    # pre_filter_mean()
    # pre_filter_plot()
    # grid_search()
    recal_score('Vistral-7B-Chat+all_filter_data_mask_half_question.csv', 0.3)
    # filter('vistral_7b_mini_logging_file_domain.csv', 0.5)
    # filter('Vistral-7B-Chat+domain_masked_wrong_answer.csv', 0.5)
    # filter('Vistral-7B-Chat+domain_masked_half_question.csv', 0.5)
    pass