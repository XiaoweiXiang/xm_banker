import os
import pickle
import random
import shutil
import numpy as np
import pandas as pd
import lightgbm as lgb
from .utils.binary_classification.modeling import evaluate_model


def create_folder(folder_dir):
    
    if os.path.exists(folder_dir):
        shutil.rmtree(path=folder_dir)
    os.makedirs(folder_dir, exist_ok=True)


def load_master(df_sample, data_f_dir, join_keys):

    df_master = df_sample
    files = list(filter(lambda x: x.split('.')[1] == 'pkl', os.listdir(data_f_dir)))
    for f in files:
        df = pickle.load(open(os.path.join(data_f_dir, f), 'rb'))
        df_master = pd.merge(df_master, df, on=join_keys, how='left')
    
    return df_master


def split_samples(df_master, split_keys, ins_size=0.8, random_seed=2022):

    df_random = df_master[split_keys].drop_duplicates().reset_index(drop=True)
    random.seed(random_seed)
    df_random['random_number'] = [random.random() for x in range(df_random.shape[0])]
    df_random['train_ind'] = df_random['random_number'] <= ins_size
    df_master = pd.merge(df_master, df_random, on=split_keys, how='inner')
    
    df_ins = df_master.loc[df_master['train_ind'], ].reset_index(drop=True).drop(columns=['random_number', 'train_ind'])
    df_oos = df_master.loc[~df_master['train_ind'], ].reset_index(drop=True).drop(columns=['random_number', 'train_ind'])

    return df_ins, df_oos


def eval_model(y_true, y_score, result_m_dir, dataname):

    df_pr = evaluate_model.calc_pr(y_true=y_true, y_score=y_score)
    df_pr['F2'] = [5 * r * p / (4 * p + r) for r,p in zip(df_pr['recall'], df_pr['precision'])]
    f2_info = df_pr.iloc[np.where(df_pr['F2'] == df_pr['F2'].max())[0], ]
    df_pr.to_csv(os.path.join(result_m_dir, '{0}_pr_threshold_{1:.5f}_f2_{2:.2f}.csv'.format(dataname, f2_info['thresholds'].iloc[0], f2_info['F2'].iloc[0])))
    evaluate_model.plot_pr_curve({'lgb': df_pr}, to_show=False, save_path=os.path.join(result_m_dir, '{0}_pr.png'.format(dataname)))


def save_lgb_model(lgb_md, result_m_dir):

    # save params
    with open(os.path.join(result_m_dir, 'lgb_params.pkl'), 'wb') as f:
        pickle.dump(lgb_md.params, f)

    # save featureimp
    lgb_md_imp = pd.DataFrame({
        'features': lgb_md.feature_name(), 
        'imp_split': lgb_md.feature_importance(importance_type='split'),
        'imp_gain': lgb_md.feature_importance(importance_type='gain')
    })
    lgb_md_imp = lgb_md_imp.sort_values(by='imp_gain', ascending=False)
    lgb_md_imp.to_csv(os.path.join(result_m_dir, 'lgb_featureimp.csv'))

    # save modelfile
    lgb_md.save_model(os.path.join(result_m_dir, 'model_file.txt'))

