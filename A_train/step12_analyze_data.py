import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

sys.path.append('.')
import config
from library.utils.statistics.descriptive import desc_dataframe

# tables
tbls = [
    # 'df_train',
    # 'df_test',
    # 'df_user_basic',
    # 'df_user_risk',
    # 'df_user_asset',
    'df_asset_info',
]

# describe
for tbl in tqdm(tbls):

    df = pickle.load(open(os.path.join(config.data_p_dir, '{0}.pkl'.format(tbl)), 'rb'))
    logging.info('>> 读取{0} shape={1}'.format(tbl, df.shape))
    logging.info('>>>> 数据示例: \n{0}'.format(df.iloc[0]))

    numcol_desc_df, charcol_desc_df = desc_dataframe.desc_dataframe(
        df=df, 
        num_metric_list=['missrate', 'unique', 'range', 'quantile'], 
        char_metric_list=['missrate', 'unique', ], 
        num_missing_values=[-1, ], 
        char_missing_values=['-1', ], 
        digit=5, 
        cn_header=False,
    )
    numcol_desc_df.to_csv(os.path.join(config.result_p_dir, '{0}_numcol_desc.csv'.format(tbl)))
    charcol_desc_df.to_csv(os.path.join(config.result_p_dir, '{0}_charcol_desc.csv'.format(tbl)))