import os
import logging
from datetime import datetime
import pandas as pd
from . import config
logging.basicConfig(level=logging.INFO, format='%(message)s')


# df_train
def load_df_train():

    df_x_train = pd.read_csv(os.path.join(config.data_o_dir, '主表数据/x_train.csv'))
    df_x_train.rename(columns={'a2': 'prod_type', 'a3': 'deal_date'}, inplace=True)
    df_x_train['prod_type'] = df_x_train['prod_type'].astype('int8')
    df_x_train['deal_date'] = pd.to_datetime(df_x_train['deal_date'])
    logging.info('df_x_train shape={0}'.format(df_x_train.shape))

    df_y_train = pd.read_csv(os.path.join(config.data_o_dir, '主表数据/y_train.csv'))
    df_y_train.loc[:, 'y'] = df_y_train.loc[:, 'y'].astype('int8')
    logging.info('df_y_train shape={0}'.format(df_y_train.shape))

    df_train = pd.merge(df_x_train, df_y_train, on=['id'], how='left')
    logging.info('df_train: ')
    df_train.info(memory_usage='deep')

    return df_train


# df_test
def load_df_test():

    df_test = pd.read_csv(os.path.join(config.data_o_dir, '主表数据/x_test.csv'))
    df_test.rename(columns={'c2': 'prod_type', 'c3': 'deal_date',}, inplace=True)
    df_test['prod_type'] = df_test['prod_type'].astype('int8')
    df_test['deal_date'] = pd.to_datetime(df_test['deal_date'])
    logging.info('df_test shape={0}'.format(df_test.shape))
    df_test.info(memory_usage='deep')

    return df_test


# df_features
# df_user_basic
def load_df_user_basic():

    df_user_basic = pd.read_csv(os.path.join(config.data_o_dir, '其他数据表/d.csv'))
    df_user_basic.rename(columns={'d1': 'sex', 'd2': 'cust_level', 'd3': 'age'}, inplace=True)
    df_user_basic[['sex', 'cust_level', 'age']] = df_user_basic[['sex', 'cust_level', 'age']].fillna(-1)
    df_user_basic[['sex', 'cust_level', 'age']] = df_user_basic[['sex', 'cust_level', 'age']].astype('int8')
    df_user_basic.info(memory_usage='deep')
    logging.info('df_userinfo shape={0}'.format(df_user_basic.shape))

    return df_user_basic

# df_user_risk
def load_df_user_risk():

    df_user_risk = pd.read_csv(os.path.join(config.data_o_dir, '其他数据表/e.csv'))
    df_user_risk.rename(columns={'e1': 'cust_risklevel', 'e2': 'eval_date'}, inplace=True)
    df_user_risk['cust_risklevel'] = df_user_risk['cust_risklevel'].astype('int8')
    df_user_risk['eval_date'] = [datetime.strptime(str(x), '%Y%m%d') for x in df_user_risk['eval_date']]
    df_user_risk.info(memory_usage='deep')
    logging.info('df_userrisk shape={0}'.format(df_user_risk.shape))

    return df_user_risk

# df_user_assets
def load_df_user_assets():

    df_user_assets = pd.read_csv(os.path.join(config.data_o_dir, '其他数据表/f.csv'))
    df_user_assets.info(memory_usage='deep')
    df_user_assets[['f'+str(x) for x in range(2, 22)]] = df_user_assets[['f'+str(x) for x in range(2, 22)]].fillna('-1')
    df_user_assets[['f'+str(x) for x in range(2, 22)]] = df_user_assets[['f'+str(x) for x in range(2, 22)]].applymap(lambda x: str(x).replace(',', ''))
    df_user_assets[['f'+str(x) for x in range(2, 22)]] = df_user_assets[['f'+str(x) for x in range(2, 22)]].astype('float')
    df_user_assets.rename(columns={
        'f1': 'cre_date', 
        'f2': 'fixed_amt_q', 'f3': 'cd_amt_q', 'f4': 'a_amt_q', 'f5': 'im_amt_q', 'f6': 'fund_amt_q', 
        'f7': 'fixed_amt', 'f8': 'cd_amt', 'f9': 'a_amt', 'f10': 'im_amt', 'f11': 'fund_amt', 
        'f12': 'fixed_amt_m', 'f13': 'cd_amt_m', 'f14': 'a_amt_m', 'f15': 'im_amt_m', 'f16': 'fund_amt_m', 
        'f17': 'fixed_amt_y', 'f18': 'cd_amt_y', 'f19': 'a_amt_y', 'f20': 'im_amt_y', 'f21': 'fund_amt_y', 
        'f22': 'dt',
        }, inplace=True)
    df_user_assets['cre_date'] = pd.to_datetime(df_user_assets['cre_date'])
    df_user_assets['dt'] = [datetime.strptime(str(x), '%Y%m%d') for x in df_user_assets['dt']]
    df_user_assets.info(memory_usage='deep')
    logging.info('df_user_assets shape={0}'.format(df_user_assets.shape))

    return df_user_assets



