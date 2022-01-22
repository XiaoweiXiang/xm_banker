import os
import sys
import pickle
import numpy as np
import lightgbm as lgb
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

sys.path.append('.')
import config
from library import model_tool

modelname = input('请为模型命名（例如：[思路]_[日期])：')
logging.info('<<<< 完成模型文件夹创建 >>>> 模型名称: {0}'.format(modelname))

result_dir = os.path.join(config.result_m_dir, 'lgb', modelname)
model_tool.create_folder(result_dir)

# load data
df_train = pickle.load(open(os.path.join(config.data_p_dir, 'df_train.pkl'), 'rb'))
df_master = model_tool.load_master(df_train, os.path.join(config.data_f_dir, 'train'), join_keys=['id'])
df_ins, df_oos = model_tool.split_samples(df_master=df_master, split_keys=['id'], ins_size=0.8, random_seed=config.random_seed)
logging.info('<<<< 完成数据读取 >>>> 总计={0}, 切分后：ins={1} {2:.2%}、oos={3} {4:.2%}'.format(df_master.shape, df_ins.shape, np.mean(df_ins.y), df_oos.shape, np.mean(df_oos.y), ))

# train model
result_m_dir = os.path.join(result_dir, 'model')
os.makedirs(result_m_dir, exist_ok=True)

features = list(filter(lambda x: x not in config.ignore_cols, df_ins.columns))
lgb_ins = lgb.Dataset(data=df_ins[features], label=df_ins['y'])
lgb_oos = lgb.Dataset(data=df_oos[features], label=df_oos['y'])

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['auc'],
    # 'max_depth': 5,
    # 'num_leaves': 8,
    'min_data_in_leaf': 10000,
    'learning_rate': 0.1,
    'feature_fraction': 0.7,
    'bagging_fraction': 1,
    'bagging_freq': 3,
    'lambda_l1': 1,
    'random_state': config.random_seed
}

bst_md = lgb.train(
    params=params, 
    train_set=lgb_ins, 
    valid_sets=[lgb_oos],
    num_boost_round=1000,
    early_stopping_rounds=5
)
logging.info('<<<< 完成模型训练 >>>>')


# eval model
df_ins.loc[:, 'score'] = bst_md.predict(df_ins.loc[:, features], num_iteration=bst_md.best_iteration)
model_tool.eval_model(y_true=df_ins['y'], y_score=df_ins['score'], result_m_dir=result_m_dir, dataname='ins')
df_oos.loc[:, 'score'] = bst_md.predict(df_oos.loc[:, features], num_iteration=bst_md.best_iteration)
model_tool.eval_model(y_true=df_oos['y'], y_score=df_oos['score'], result_m_dir=result_m_dir, dataname='oos')
logging.info('<<<< 完成模型效果评估 >>>>')


# save model
model_tool.save_lgb_model(lgb_md=bst_md, result_m_dir=result_m_dir)
logging.info('<<<< 完成模型结果保存 >>>>')
