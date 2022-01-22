import os
import logging
from datetime import datetime
import pandas as pd
from library.utils.datautils import dataframe_utils
logging.basicConfig(level=logging.INFO, format='%(message)s')


# df_train
def load_df_train(data_dir):

    df_x_train = pd.read_csv(os.path.join(data_dir, '主表数据/x_train.csv'))
    df_x_train['a3'] = pd.to_datetime(df_x_train['a3'])
    df_x_train.rename(columns={'a2': 'prod_type', 'a3': 'deal_date',}, inplace=True)
    logging.info('df_x_train shape={0}'.format(df_x_train.shape))

    df_y_train = pd.read_csv(os.path.join(data_dir, '主表数据/y_train.csv'))
    df_y_train['y'] = df_y_train['y'].astype('int8')
    logging.info('df_y_train shape={0}'.format(df_y_train.shape))

    df_train = pd.merge(df_x_train, df_y_train, on=['id'], how='left')
    df_train = dataframe_utils.downcast_df(df_train)
    logging.info('df_train shape={0}'.format(df_train.shape))
    df_train.info(memory_usage='deep')

    return df_train


# df_test
def load_df_test(data_dir):

    df_test = pd.read_csv(os.path.join(data_dir, '主表数据/x_test.csv'))
    df_test['c3'] = pd.to_datetime(df_test['c3'])
    df_test.rename(columns={'c2': 'prod_type', 'c3': 'deal_date',}, inplace=True)
    df_test = dataframe_utils.downcast_df(df_test)
    logging.info('df_test shape={0}'.format(df_test.shape))
    df_test.info(memory_usage='deep')

    return df_test


# df_user_basic
def load_df_user_basic(data_dir):

    df_user_basic = pd.read_csv(os.path.join(data_dir, '其他数据表/d.csv'))
    df_user_basic.rename(columns={'d1': 'sex', 'd2': 'cust_level', 'd3': 'age'}, inplace=True)
    df_user_basic = dataframe_utils.downcast_df(df_user_basic)
    logging.info('df_userinfo shape={0}'.format(df_user_basic.shape))
    df_user_basic.info(memory_usage='deep')

    return df_user_basic


# df_user_risk
def load_df_user_risk(data_dir):

    df_user_risk = pd.read_csv(os.path.join(data_dir, '其他数据表/e.csv'))
    df_user_risk['e2'] = [datetime.strptime(str(x), '%Y%m%d') for x in df_user_risk['e2']]
    df_user_risk.rename(columns={'e1': 'risk_level', 'e2': 'eval_date'}, inplace=True)
    df_user_risk = dataframe_utils.downcast_df(df_user_risk)
    logging.info('df_userrisk shape={0}'.format(df_user_risk.shape))
    df_user_risk.info(memory_usage='deep')

    return df_user_risk


# df_user_asset
def load_df_user_asset(data_dir):

    df_user_asset = pd.read_csv(os.path.join(data_dir, '其他数据表/f.csv'))
    df_user_asset['f1'] = pd.to_datetime(df_user_asset['f1'])
    df_user_asset[['f'+str(x) for x in range(2, 22)]] = df_user_asset[['f'+str(x) for x in range(2, 22)]]
    df_user_asset[['f'+str(x) for x in range(2, 22)]] = df_user_asset[['f'+str(x) for x in range(2, 22)]].applymap(lambda x: str(x).replace(',', ''))
    df_user_asset[['f'+str(x) for x in range(2, 22)]] = df_user_asset[['f'+str(x) for x in range(2, 22)]].astype('float')
    df_user_asset['f22'] = [datetime.strptime(str(x), '%Y%m%d') for x in df_user_asset['f22']]
    df_user_asset.rename(columns={
        'f1': 'reg_date', 
        'f2': 'fixed_amt_q', 'f3': 'cd_amt_q', 'f4': 'a_amt_q', 'f5': 'im_amt_q', 'f6': 'fund_amt_q', 
        'f7': 'fixed_amt', 'f8': 'cd_amt', 'f9': 'a_amt', 'f10': 'im_amt', 'f11': 'fund_amt', 
        'f12': 'fixed_amt_m', 'f13': 'cd_amt_m', 'f14': 'a_amt_m', 'f15': 'im_amt_m', 'f16': 'fund_amt_m', 
        'f17': 'fixed_amt_y', 'f18': 'cd_amt_y', 'f19': 'a_amt_y', 'f20': 'im_amt_y', 'f21': 'fund_amt_y', 
        'f22': 'dt',
        }, inplace=True)
    df_user_asset = dataframe_utils.downcast_df(df_user_asset)
    logging.info('df_user_assets shape={0}'.format(df_user_asset.shape))
    df_user_asset.info(memory_usage='deep')

    return df_user_asset


# df_asset_info
def load_df_asset_info(data_dir): 
    
    df_asset_a1 = load_df_asset_info_a1(data_dir)
    df_asset_b1 = load_df_asset_info_b1(data_dir)
    df_asset_c1 = load_df_asset_info_c1(data_dir)
    df_asset_d1 = load_df_asset_info_d1(data_dir)
    df_asset_info = pd.concat([df_asset_a1, df_asset_b1, df_asset_c1, df_asset_d1], axis=0, join='outer')
    cols = [
        'prod_code', 
        'prod_type_code', 
        'prod_category', 
        'prod_model', 
        'prod_model2', 
        'prod_risk_level', 
        'prod_price', 
        'prod_cycle', 
        'prod_hold_days', 
        'prod_has_rollover', 
        'prod_allowed_change_dividend', 
        'prod_share_frozen_rate', 
        'prod_face_value', 
        'prod_issue_price', 
        'prod_return_base', 
        'prod_return_rate', 
        'prod_exp_return_rate_min', 
        'prod_exp_return_rate_max', 
        'dt', 
    ]
    df_asset_info = df_asset_info[cols]
    df_asset_info = dataframe_utils.downcast_df(df_asset_info)
    logging.info('df_asset_info shape={0}'.format(df_asset_info.shape))
    df_asset_info.info(memory_usage='deep')

    return df_asset_info

# df_asset_info_a1
def load_df_asset_info_a1(data_dir):

    df_asset = pd.read_csv(os.path.join(data_dir, '其他数据表/g.csv'))
    # df_asset['g9'] = [datetime.strptime(str(x), '%y%m') for x in df_asset['g9']]
    df_asset.rename(columns={
        'g1': 'prod_price',
        'g2': 'prod_cycle',
        'g3': 'prod_model',
        'g4': 'prod_risk_level',
        'g5': 'prod_has_rollover',
        'g6': 'prod_allowed_change_dividend',
        'g7': 'prod_category',
        'g8': 'prod_hold_days',
        'g9': 'dt',
    }, inplace=True)
    df_asset['prod_type_code'] = 1
    df_asset = dataframe_utils.downcast_df(df_asset)
    logging.info('df_asset_a1 shape={0}'.format(df_asset.shape))
    df_asset.info(memory_usage='deep')
    
    return df_asset

# df_asset_info_b1
def load_df_asset_info_b1(data_dir):

    df_asset = pd.read_csv(os.path.join(data_dir, '其他数据表/h.csv'))
    # df_asset['h8'] = [datetime.strptime(str(x), '%y%m') for x in df_asset['h8']]
    df_asset.rename(columns={
        'h1': 'prod_price',
        'h2': 'prod_cycle',
        'h3': 'prod_model',
        'h4': 'prod_risk_level',
        'h5': 'prod_allowed_change_dividend',
        'h6': 'prod_category',
        'h7': 'prod_model2',
        'h8': 'dt',
    }, inplace=True)
    df_asset['prod_type_code'] = 2
    df_asset = dataframe_utils.downcast_df(df_asset)
    logging.info('df_asset_b1 shape={0}'.format(df_asset.shape))
    df_asset.info(memory_usage='deep')

    return df_asset

# df_asset_info_c1
def load_df_asset_info_c1(data_dir):

    df_asset = pd.read_csv(os.path.join(data_dir, '其他数据表/i.csv'))
    # df_asset['i9'] = [datetime.strptime(str(x), '%y%m') for x in df_asset['i9']]
    df_asset.rename(columns={
        'i1': 'prod_price',
        'i2': 'prod_cycle',
        'i3': 'prod_model',
        'i4': 'prod_risk_level',
        'i5': 'prod_allowed_change_dividend',
        'i6': 'prod_share_frozen_rate',
        'i7': 'prod_category',
        'i8': 'prod_hold_days',
        'i9': 'dt',
    }, inplace=True)
    df_asset['prod_type_code'] = 3
    df_asset = dataframe_utils.downcast_df(df_asset)
    logging.info('df_asset_c1 shape={0}'.format(df_asset.shape))
    df_asset.info(memory_usage='deep')

    return df_asset

# df_asset_info_d1
def load_df_asset_info_d1(data_dir):

    df_asset = pd.read_csv(os.path.join(data_dir, '其他数据表/j.csv'))
    df_asset['j11'] = df_asset['j11'] / 100
    # # df_asset['j13'] = [datetime.strptime(str(x), '%y%m') for x in df_asset['j13']]
    df_asset.rename(columns={
        'j1': 'prod_price',
        'j2': 'prod_cycle',
        'j3': 'prod_model',
        'j4': 'prod_risk_level',
        'j5': 'prod_allowed_change_dividend',
        'j6': 'prod_return_base',
        'j7': 'prod_exp_return_rate_min',
        'j8': 'prod_exp_return_rate_max',
        'j9': 'prod_face_value',
        'j10': 'prod_issue_price',
        'j11': 'prod_return_rate',
        'j12': 'prod_hold_days',
        'j13': 'dt',
    }, inplace=True)
    df_asset['prod_type_code'] = 4
    df_asset = dataframe_utils.downcast_df(df_asset)
    logging.info('df_asset_c1 shape={0}'.format(df_asset.shape))
    df_asset.info(memory_usage='deep')

    return df_asset

# df_asset_info_a2
def load_df_asset_info_a2(data_dir):

    df_asset = pd.read_csv(os.path.join(data_dir, '其他数据表/k.csv'))
    # df_asset[['g'+str(x) for x in range(1, 8)]] = df_asset[['g'+str(x) for x in range(1, 8)]].astype('int8')
    # df_asset['g8'] = df_asset['g8'].astype('int16')
    # df_asset.rename(columns={
        # 'g1': 'prod_price',
        # 'g2': 'prod_cycle',
        # 'g3': 'prod_model',
        # 'g4': 'prod_risk_level',
        # 'g5': 'prod_has_rollover',
        # 'g6': 'prod_allowed_change_dividend',
        # 'g7': 'prod_category',
        # 'g8': 'prod_hold_days',
        # 'g9': 'dt',
    # }, inplace=True)
    # df_asset['prod_type_code'] = 1
    # print(df_asset.sort_values(by=['dt']).head())
    # df_asset['dt'] = [datetime.strptime(str(x), '%y%m') for x in df_asset['dt']]
    logging.info('df_asset_a1 shape={0}'.format(df_asset.shape))
    df_asset.info(memory_usage='deep')
    print(df_asset.head())
    print(df_asset.describe())

    return df_asset






# df_asset_info_a2
def load_df_assets_a2(data_dir):

    df_assets_a = pd.read_csv(os.path.join(data_dir, '其他数据表/k.csv'))
    # df_assets_a['dt'] = [datetime.strptime(str(x), '%y%m%d') for x in df_assets_a['dt']]
    logging.info('df_assets_a2 shape={0}'.format(df_assets_a.shape))

    return df_assets_a


# df_trade
def load_df_trade_a(data_dir):

    df_trade = pd.read_csv(os.path.join(data_dir, '其他数据表/n.csv'))
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


def load_df_trade_b(data_dir):

    df_trade = pd.read_csv(os.path.join(data_dir, '其他数据表/o.csv'))
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


def load_df_trade_c(data_dir):

    df_trade = pd.read_csv(os.path.join(data_dir, '其他数据表/q.csv'))
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


def load_df_trade_d(data_dir):

    df_trade = pd.read_csv(os.path.join(data_dir, '其他数据表/p.csv'))
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


def load_df_trade(data_dir):

    df_trade_a = load_df_trade_a()
    df_trade_b = load_df_trade_b()
    df_trade_c = load_df_trade_c()
    df_trade_d = load_df_trade_d()
    df_trade = pd.concat([df_trade_a, df_trade_b, df_trade_c, df_trade_d], axis=0, join='outer')
    print(df_trade.describe())
    print(df_trade.head())
    df_trade.info(memory_usage='deep')

    return df_trade

