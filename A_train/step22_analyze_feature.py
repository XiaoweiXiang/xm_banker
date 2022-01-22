import os
import sys
import pickle
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

sys.path.append('.')
import config
from library import model_tool
from library.utils.binary_classification.var_analysis.VarStats import *


# tables
tbls = [
    # 'feature_u_info',
    'feature_u_risk',
    'feature_u_asset',
    'feature_p_info',
    'feature_u_trade',
    'feature_u_p_trade',
    # 'feature_app',
    # 'feature_basic',
]

# load data
df_train = pickle.load(open(os.path.join(config.data_p_dir, 'df_train.pkl'), 'rb'))
logging.info('<<<< 完成样本数据读取 >>>> 总计={0}'.format(df_train.shape))

for tbl in tbls:
    analysis_path = os.path.join(config. result_f_dir, tbl)
    model_tool.create_folder(analysis_path)
    
    df_tbl = pickle.load(open(os.path.join(config.data_f_dir, 'train', '{0}.pkl'.format(tbl)), 'rb'))
    logging.info('<<<< 完成特征数据读取 >>>> 表名称：{0}, 总计={1}'.format(tbl, df_tbl.shape))
    df_master = pd.merge(df_train, df_tbl, on=['id'], how='left')
    df_master['deal_date'] = df_master['deal_date'].astype('str')

    var_stats = VarStats(
        df_master=df_master, 
        ignorevar_names=config.ignore_cols, 
        analysis_path=analysis_path, 
        digit=3, 
        datetime_name=None, 
        oot_start_dt=None, 
        group_name='deal_date', 
        to_save=True
    )

    var_stats.calc_numvars_stats(
        num_metric_list=['missrate', 'unique', 'range', 'quantile'], 
        num_missing_values=[-1, -999], 
        digit=5, 
        cn_header=False
    )

    var_stats.calc_numvars_iv(
        tgt_name='y', 
        spec_values=[-1, -999], 
        max_bins=5, 
        min_prop_in_bin=0.05, 
        equi_method='equif', 
        equi_bins=100, 
        binning_criteria='chi2', 
        plot_woe=True, 
        show_woe=False, 
        save_woe=True
    )