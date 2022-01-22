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
    # '主表数据/x_train',
    # '主表数据/y_train',
    # '主表数据/x_test',
    # '其他数据表/d',
    # '其他数据表/e',
    # '其他数据表/f',
    # '其他数据表/g',
    # '其他数据表/h',
    # '其他数据表/i',
    # '其他数据表/j',
    # '其他数据表/k',
    # '其他数据表/l',
    # '其他数据表/m',
    # '其他数据表/n',
    # '其他数据表/o',
    # '其他数据表/p',
    # '其他数据表/q',
    # '其他数据表/r',
    '其他数据表/s',
]

# describe
for tbl in tqdm(tbls):

    df = pd.read_csv(open(os.path.join(config.data_o_dir, '{0}.csv'.format(tbl)), 'rb'))
    logging.info('>> 读取{0} shape={1}'.format(tbl, df.shape))

    numcol_desc_df, charcol_desc_df = desc_dataframe.desc_dataframe(
        df=df, 
        num_metric_list=['missrate', 'unique', 'range', 'quantile'], 
        char_metric_list=['missrate', 'unique', ], 
        num_missing_values=[-1, ], 
        char_missing_values=['-1', ], 
        digit=5, 
        cn_header=False,
    )
    desc_df = pd.concat([numcol_desc_df, charcol_desc_df], join='outer').loc[df.columns, ]
    example = df.iloc[0]
    example.name = 'example'
    desc_df = desc_df.join(example)
    desc_df.to_csv(os.path.join(config.result_o_dir, '{0}_desc.csv'.format(tbl.split('/')[1])))