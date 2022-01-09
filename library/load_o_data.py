import os
import logging
from datetime import datetime
import pandas as pd
from . import config
# import config
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


# df_assets_a
def load_df_assets_a1():

    df_assets_a = pd.read_csv(os.path.join(config.data_o_dir, '其他数据表/g.csv'))
    df_assets_a[['g'+str(x) for x in range(1, 7)]] = df_assets_a[['g'+str(x) for x in range(1, 7)]].astype('int8')
    df_assets_a.rename(columns={
        'g1': 'price', 
        'g2': 'cycle', 
        'g3': 'model', 
        'g4': 'risk_level', 
        'g5': 'has_rollover', 
        'g6': 'allowed_change_dividend', 
        'g7': 'prod_type', 
        'g8': 'hold_days', 
        'g9': 'dt',
    }, inplace=True)
    # df_assets_a['dt'] = [datetime.strptime(str(x), '%y%m%d') for x in df_assets_a['dt']]
    logging.info('df_assets_a1 shape={0}'.format(df_assets_a.shape))

    return df_assets_a


# df_assets_a2
def load_df_assets_a2():

    df_assets_a = pd.read_csv(os.path.join(config.data_o_dir, '其他数据表/k.csv'))
    df_assets_a[['k'+str(x) for x in range(1, 11)]] = df_assets_a[['k'+str(x) for x in range(1, 11)]].fillna(-1)
    df_assets_a[['k'+str(x) for x in range(1, 6)]] = df_assets_a[['k'+str(x) for x in range(1, 6)]].astype('int8')
    df_assets_a[['k10']] = df_assets_a[['k10']].astype('int8')
    # df_assets_a['dt'] = [datetime.strptime(str(x), '%y%m%d') for x in df_assets_a['dt']]
    logging.info('df_assets_a2 shape={0}'.format(df_assets_a.shape))

    return df_assets_a

# df_assets
def load_df_assets(): 
    
    df_assets_a1 = load_df_assets_a1()
    df_assets_a2 = load_df_assets_a2()
    df_assets = pd.merge(df_assets_a2, df_assets_a1, how='outer', on=['prod_code'])

    logging.info('df_assets shape={0}'.format(df_assets.shape))

    return df_assets


# df_trade
def load_df_trade_a():

    df_trade = pd.read_csv(os.path.join(config.data_o_dir, '其他数据表/n.csv'))
    df_trade.info(memory_usage='deep')
    df_trade[['n2']] = df_trade[['n2']].fillna(-1)
    df_trade[['n7']] = df_trade[['n7']].applymap(lambda x: str(x).replace(',', ''))
    df_trade[['n2', 'n3', 'n8', 'n9']] = df_trade[['n2', 'n3', 'n8', 'n9']].astype('int8')
    df_trade[['n6', 'n7', 'n10']] = df_trade[['n6', 'n7', 'n10']].astype('float')
    df_trade['n11'] = [datetime.strptime(str(x), '%Y%m%d') for x in df_trade['n11']]
    df_trade.rename(columns={
        'n1': 'trade_id',
        'n2': 'trade_code',
        'n3': 'trade_channel',
        'core_cust_id': 'core_cust_id',
        'prod_code': 'prod_code',
        'n6': 'trade_net_value',
        'n7': 'trade_apply_amt',
        'n8': 'trade_fund_status',
        'n9': 'trade_status',
        'n10': 'trade_total_amt',
        'n11': 'deal_date',
    }, inplace=True)
    print(df_trade.describe())
    df_trade.info(memory_usage='deep')

    return df_trade


def load_df_trade_b():

    df_trade = pd.read_csv(os.path.join(config.data_o_dir, '其他数据表/o.csv'))
    df_trade.info(memory_usage='deep')
    df_trade[['o2', 'o3', 'o8', 'o9']] = df_trade[['o2', 'o3', 'o8', 'o9']].astype('int8')
    df_trade[['o7', 'o10', 'o11']] = df_trade[['o7', 'o10', 'o11']].applymap(lambda x: str(x).replace(',', '')).astype('float')
    df_trade['o12'] = [datetime.strptime(str(x), '%Y%m%d') for x in df_trade['o12']]
    df_trade.rename(columns={
        'o1': 'trade_id',
        'o2': 'trade_code',
        'o3': 'trade_channel',
        'core_cust_id': 'core_cust_id',
        'prod_code': 'prod_code',
        'o6': 'trade_net_value',
        'o7': 'trade_apply_amt',
        'o8': 'trade_status',
        'o9': 'trade_fund_status',
        'o10': 'trade_total_amt',
        'o11': 'trade_excess_amt',
        'o12': 'deal_date',
    }, inplace=True)
    print(df_trade.head())
    print(df_trade.describe())
    df_trade.info(memory_usage='deep')

    return df_trade


def load_df_trade_c():

    df_trade = pd.read_csv(os.path.join(config.data_o_dir, '其他数据表/q.csv'))
    df_trade.info(memory_usage='deep')
    df_trade[['q7']] = df_trade[['q7']].applymap(lambda x: str(x).replace(',', ''))
    df_trade[['q6', 'q7']] = df_trade[['q6', 'q7']].astype('float')
    df_trade[['q2', 'q3', 'q8', 'q9']] = df_trade[['q2', 'q3', 'q8', 'q9']].astype('int8')
    df_trade['q10'] = [datetime.strptime(str(x), '%Y%m%d') for x in df_trade['q10']]
    df_trade.rename(columns={
        'q1': 'trade_id',
        'q2': 'trade_code',
        'q3': 'trade_channel',
        'core_cust_id': 'core_cust_id',
        'prod_code': 'prod_code',
        'q6': 'trade_net_value',
        'q7': 'trade_apply_amt',
        'q8': 'trade_fund_status',
        'q9': 'trade_status',
        'q10': 'deal_date',
    }, inplace=True)
    df_trade.info(memory_usage='deep')

    return df_trade


def load_df_trade_d():

    df_trade = pd.read_csv(os.path.join(config.data_o_dir, '其他数据表/p.csv'))
    df_trade.info(memory_usage='deep')
    df_trade[['p3', 'p10']] = df_trade[['p3', 'p10']].fillna(-1)
    df_trade[['p7']] = df_trade[['p7']].applymap(lambda x: str(x).replace(',', ''))
    df_trade[['p6', 'p7', 'p8']] = df_trade[['p6', 'p7', 'p8']].astype('float')
    df_trade[['p2', 'p3', 'p9', 'p10', 'p11']] = df_trade[['p2', 'p3', 'p9', 'p10', 'p11']].astype('int8')
    df_trade['p12'] = [datetime.strptime(str(x), '%Y%m%d') for x in df_trade['p12']]
    df_trade.rename(columns={
        'p1': 'trade_id',
        'p2': 'trade_code',
        'p3': 'trade_channel',
        'core_cust_id': 'core_cust_id',
        'prod_code': 'prod_code',
        'p6': 'trade_net_value',
        'p7': 'trade_apply_amt',
        'p8': 'trade_discount_rate',
        'p9': 'trade_status',
        'p10': 'trade_fund_status',
        'p11': 'trade_fee_rate',
        'p12': 'deal_date',
    }, inplace=True)
    df_trade.info(memory_usage='deep')

    return df_trade


def load_df_trade():

    df_trade_a = load_df_trade_a()
    df_trade_b = load_df_trade_b()
    df_trade_c = load_df_trade_c()
    df_trade_d = load_df_trade_d()
    df_trade = pd.concat([df_trade_a, df_trade_b, df_trade_c, df_trade_d], axis=0, join='outer')
    print(df_trade.describe())
    print(df_trade.head())
    df_trade.info(memory_usage='deep')

    return df_trade

