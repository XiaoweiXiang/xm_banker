import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

sys.path.append('.')
import config
from library import model_tool

modelname = input('请输入模型名称：')
threshold = float(input('请输入切分阈值：'))

result_m_dir = os.path.join(config.result_m_dir, 'lgb', modelname, 'model')
result_p_dir = os.path.join(config.result_m_dir, 'lgb', modelname, 'predict')
os.makedirs(result_p_dir, exist_ok=True)

# load data
df_test = pickle.load(open(os.path.join(config.data_p_dir, 'df_test.pkl'), 'rb'))
df_master = model_tool.load_master(df_test, os.path.join(config.data_f_dir, 'test'), join_keys=['id'])
logging.info('<<<< 完成数据读取 >>>> 总计={0}'.format(df_master.shape))

# load model
md = lgb.Booster(model_file=os.path.join(result_m_dir, 'model_file.txt'))

# predict
df_master['score'] = md.predict(df_master[md.feature_name()], num_iteration=md.best_iteration)
df_master['y'] = [int(x>threshold) for x in df_master['score']]
logging.info('<<<< 完成模型打分 >>>>')

# save result
df_master[['id', 'y']].to_csv(os.path.join(result_p_dir, 'lgb_{0}_threshold_{1}.csv'.format(modelname, threshold)), index=False)
logging.info('<<<< 完成文件保存 >>>>')
