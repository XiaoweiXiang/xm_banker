from library.utils.statistics.descriptive import desc_dataframe
import os
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import date, timedelta
logging.basicConfig(level=logging.INFO, format='%(message)s')

__all__ = [
    'proc_u_info',
    'proc_u_risk',
    'proc_u_trade',
    'proc_u_p_trade',
]

def downcast_df(df):

    float_cols = df.select_dtypes('float64').columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
    int_cols = df.select_dtypes('int64').columns
    print(int_cols)
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')

    return df


def proc_u_info(df_sample, data_dir):

    df_user_basic = pickle.load(open(os.path.join(data_dir, 'df_user_basic.pkl'), 'rb'))
    
    logging.info('\n<< 用户/客户信息特征: 开始计算 >>')
    u_info = pd.merge(df_sample[['id', 'core_cust_id']], df_user_basic, on=['core_cust_id'], how='left')
    u_info = u_info.drop(columns=['core_cust_id'])
    u_info = u_info.fillna(-1)
    u_info.columns = ['_'.join(['u_info', x]) if x != 'id' else x for x in u_info.columns]
    logging.info('<< 用户/客户信息特征: 完成计算 feature_u_info={0} >>'.format(u_info.shape))
    u_info.info(memory_usage='deep')

    return u_info


def proc_u_risk(df_sample, data_dir, max_delta = 6):

    df_user_risk = pickle.load(open(os.path.join(data_dir, 'df_user_risk.pkl'), 'rb'))
    df_user_risk_tmp = pd.merge(df_sample, df_user_risk, on=['core_cust_id'], how='inner')

    logging.info('\n<< 用户/风险评估行为信息特征: 开始计算 >>')
    df_user_risk_tmp = __clean_u_risk(df_user_risk_tmp)
    u_risk = __calc_u_risk(df_user_risk_tmp, max_delta)
    u_risk = u_risk.fillna(0)
    u_risk = pd.merge(df_sample[['id']], u_risk, on=['id'], how='left')
    u_risk.columns = ['_'.join(['u_risk', x]) if x != 'id' else x for x in u_risk.columns]
    u_risk = u_risk.fillna(-1) # 未评估
    u_risk = downcast_df(u_risk)
    logging.info('<< 用户/风险评估行为信息特征: 完成计算 feature_u_risk={0} >>'.format(u_risk.shape))
    u_risk.info(memory_usage='deep')

    return u_risk 


def __clean_u_risk(df_user_risk_tmp):

    df_user_risk_tmp = df_user_risk_tmp.loc[df_user_risk_tmp['eval_date'] < df_user_risk_tmp['deal_date'], ].reset_index(drop=True)
    df_user_risk_tmp['eval_cnt'] = 1
    df_user_risk_tmp['elapsed_day'] = [x.days for x in df_user_risk_tmp['deal_date'] - df_user_risk_tmp['eval_date']]
    df_user_risk_tmp['elapsed_month'] = [12*(x.year - y.year) + (x.month - y.month) for x,y in zip(df_user_risk_tmp['deal_date'], df_user_risk_tmp['eval_date'])]
    df_user_risk_tmp['elapsed_order'] = df_user_risk_tmp.groupby(['id'])['elapsed_day'].rank()
    logging.info('<<<< 清洗用户风险评估数据 {0} >>>>'.format(df_user_risk_tmp.shape))

    return df_user_risk_tmp


def __calc_u_risk(df_user_risk_tmp, max_delta):

    his = __calc_u_risk_his(df_user_risk_tmp)
    lst = __calc_u_risk_lst(df_user_risk_tmp, max_delta)
    pst = __calc_u_risk_pst(df_user_risk_tmp, max_delta)
    level = __calc_u_risk_level(df_user_risk_tmp)

    u_risk = pd.merge(his, lst, on=['id'], how='left')
    u_risk = pd.merge(u_risk, pst, on=['id'], how='left')
    u_risk = pd.merge(u_risk, level, on=['id'], how='left')

    return u_risk


def __calc_u_risk_his(df_user_risk_tmp):

    his = df_user_risk_tmp.groupby(['id']).agg({
        'elapsed_day': [max, min],
        'eval_cnt': [sum],
        'risk_level': [np.mean, max, min],
    })
    values_col = ['_'.join([x[0], x[1]]) for x in his.columns]
    his.columns = values_col
    his = his.reset_index()
    logging.info('<<<< 计算历史时段特征 {0} >>>>'.format(his.shape))

    return his


def __calc_u_risk_lst(df_user_risk_tmp, max_delta):

    lst_m_g = df_user_risk_tmp.groupby(['id', 'elapsed_month']).agg({
        'eval_cnt': [sum],
        'risk_level': [np.mean, max, min],
    })
    values_col = ['_'.join([x[0], x[1]]) for x in lst_m_g.columns]
    lst_m_g.columns = values_col
    lst_m_g = lst_m_g.reset_index()
    lst_m_g = lst_m_g.loc[lst_m_g['elapsed_month'] <= max_delta]

    lst = pd.pivot_table(lst_m_g, index=['id'], columns='elapsed_month', values=values_col, fill_value=0)
    vars_col = ['_'.join([x[0], 'l'+str(x[1])+'m']) for x in lst.columns]
    lst.columns = vars_col
    lst = lst.reset_index()
    logging.info('<<<< 计算最近第1-{0}月特征 {1} >>>>'.format(max_delta, lst.shape))

    return lst


def __calc_u_risk_pst(df_user_risk_tmp, max_delta):

    pst = df_user_risk_tmp[['id']].drop_duplicates().reset_index(drop=True)
    for m in range(1, max_delta+1):
        pst_m_g = df_user_risk_tmp.loc[df_user_risk_tmp['elapsed_month'] <= m].groupby(['id']).agg({
            'eval_cnt': [sum],
            'risk_level': [np.mean, max, min],
        })
        vars_col = ['_'.join([x[0], x[1], 'p'+str(m)+'m']) for x in pst_m_g.columns]
        pst_m_g.columns = vars_col
        pst_m_g = pst_m_g.reset_index()
        pst = pd.merge(pst, pst_m_g, on=['id'], how='left')

    for t in range(1, max_delta+1):
        pst_t_g = df_user_risk_tmp.loc[df_user_risk_tmp['elapsed_order'] <= t].groupby(['id']).agg({
            'risk_level': [np.mean, max, min],
        })
        vars_col = ['_'.join([x[0], x[1], 'p'+str(t)+'t']) for x in pst_t_g.columns]
        pst_t_g.columns = vars_col
        pst_t_g = pst_t_g.reset_index()
        pst = pd.merge(pst, pst_t_g, on=['id'], how='left')

    logging.info('<<<< 统计最近1-{0}月内特征 {1} >>>>'.format(max_delta, pst.shape))

    return pst


def __calc_u_risk_level(df_user_risk_tmp):

    level_g = df_user_risk_tmp.groupby(['id', 'risk_level']).agg({'elapsed_day': [max, min]})
    values_col = ['_'.join([x[0], x[1]]) for x in level_g.columns]
    level_g.columns = values_col
    level_g = level_g.reset_index()

    level = pd.pivot_table(level_g, index=['id'], columns='risk_level', values=values_col, fill_value=0)
    vars_col = ['_'.join([str(int(x[1])), x[0]]) for x in level.columns]
    level.columns = vars_col
    level = level.reset_index()
    logging.info('<<<< 计算风险等级特征 {0} >>>>'.format(level.shape))

    return level


def proc_u_asset(df_sample, data_dir):

    df_user_asset = pickle.load(open(os.path.join(data_dir, 'df_user_asset.pkl'), 'rb'))
    df_user_asset['deal_date'] = [(x + timedelta(days=5)).replace(day=1) for x in df_user_asset['dt']]
    df_user_asset_tmp = pd.merge(df_sample[['id', 'core_cust_id', 'deal_date']], df_user_asset, on=['core_cust_id', 'deal_date'], how='left')
    
    logging.info('\n<< 用户/资产信息特征: 开始计算 >>')
    df_user_asset_tmp['reg_elapsed_day'] = [x.days for x in df_user_asset_tmp['deal_date'] - df_user_asset_tmp['reg_date']]
    u_asset = df_user_asset_tmp.drop(columns=['core_cust_id', 'deal_date', 'reg_date', 'dt'])
    u_asset = u_asset.fillna(-1)
    u_asset.columns = ['_'.join(['u_asset', x]) if x != 'id' else x for x in u_asset.columns]
    u_asset = downcast_df(u_asset)
    logging.info('<< 用户/资产信息特征: 完成计算 feature_u_asset={0} >>'.format(u_asset.shape))
    u_asset.info(memory_usage='deep')

    return u_asset


def proc_p_info(df_sample, data_dir):

    df_asset_info = pickle.load(open(os.path.join(data_dir, 'df_asset_info.pkl'), 'rb'))
    df_asset_info_tmp = pd.merge(df_sample[['id', 'prod_code']], df_asset_info, on=['prod_code'], how='left')
    
    logging.info('\n<< 理财产品/产品信息特征: 开始计算 >>')
    p_info = df_asset_info_tmp.drop(columns=['prod_code'])
    p_info = p_info.fillna(-1)
    p_info.columns = ['_'.join(['p_info', x]) if x != 'id' else x for x in p_info.columns]
    p_info = downcast_df(p_info)
    logging.info('<< 理财产品/产品信息特征: 完成计算 feature_p_info={0} >>'.format(p_info.shape))
    p_info.info(memory_usage='deep')

    return p_info


def proc_u_trade(df_sample, data_dir):

    df_trade = pickle.load(open(os.path.join(data_dir, 'df_trade.pkl'), 'rb'))
    df_trade_tmp = pd.merge(df_sample, df_trade, on=['core_cust_id'], how='inner', suffixes=('', '_t'))
    
    logging.info('\n<< 用户_交易流水特征: 开始计算 >>')
    df_trade_tmp = __clean_trade(df_trade_tmp)
    u_trade = __calc_trade(df_trade_tmp)
    u_trade.columns = ['_'.join(['u_trade', x]) if x != 'id' else x for x in u_trade.columns]
    u_trade = u_trade.fillna(0)

    u_trade = pd.merge(df_sample[['id']], u_trade, on=['id'], how='left')
    u_trade = u_trade.fillna(-1) # 完全无trade
    u_trade = downcast_df(u_trade)
    logging.info('<< 用户_交易流水特征: 完成计算 feature_u_trade={0} >>'.format(u_trade.shape))
    u_trade.info(memory_usage='deep')

    return u_trade
    

def proc_u_p_trade(df_sample, data_dir):

    df_trade = pickle.load(open(os.path.join(data_dir, 'df_trade.pkl'), 'rb'))
    df_trade_tmp = pd.merge(df_sample, df_trade, on=['core_cust_id', 'prod_code'], how='inner', suffixes=('', '_t'))

    logging.info('\n<< 用户x理财产品_交易流水特征: 开始计算 >>')
    df_trade_tmp = __clean_trade(df_trade_tmp)
    u_p_trade = __calc_trade(df_trade_tmp)
    u_p_trade.columns = ['_'.join(['u_p_trade', x]) if x != 'id' else x for x in u_p_trade.columns]
    u_p_trade = u_p_trade.fillna(0)

    u_p_trade = pd.merge(df_sample[['id']], u_p_trade, on=['id'], how='left')
    u_p_trade = u_p_trade.fillna(-1) # 完全无trade
    u_p_trade = downcast_df(u_p_trade)
    logging.info('<< 用户x理财产品_交易流水特征: 完成计算 feature_u_p_trade={0} >>'.format(u_p_trade.shape))
    u_p_trade.info(memory_usage='deep')

    return u_p_trade


def __clean_trade(df_trade_tmp):

    df_trade_tmp = df_trade_tmp.loc[df_trade_tmp['deal_date_t'] < df_trade_tmp['deal_date'], ].reset_index(drop=True)
    df_trade_tmp['trade_cnt'] = 1
    df_trade_tmp['elapsed_day'] = [x.days for x in df_trade_tmp['deal_date'] - df_trade_tmp['deal_date_t']]
    df_trade_tmp['elapsed_month'] = [12*(x.year - y.year) + (x.month - y.month) for x,y in zip(df_trade_tmp['deal_date'], df_trade_tmp['deal_date_t'])]
    logging.info('<<<< 清洗交易数据 {0} >>>>'.format(df_trade_tmp.shape))

    return df_trade_tmp


def __calc_trade(df_trade_tmp):
    
    his = __calc_trade_his(df_trade_tmp)
    lst = __calc_trade_lst(df_trade_tmp)
    pst = __calc_trade_pst(df_trade_tmp)

    df_trade_features = pd.merge(his, lst, on=['id'], how='left')
    df_trade_features = pd.merge(df_trade_features, pst, on=['id'], how='left')

    return df_trade_features


def __calc_trade_his(df_trade_tmp):

    his_features = df_trade_tmp.groupby(['id']).agg(
        elapsed_day_min = pd.NamedAgg(column='elapsed_day', aggfunc='min'),
        elapsed_day_max = pd.NamedAgg(column='elapsed_day', aggfunc='max'),
        cnt = pd.NamedAgg(column='trade_cnt', aggfunc='sum'),
        apply_amt_sum = pd.NamedAgg(column='trade_apply_amt', aggfunc='sum'),
        apply_amt_avg = pd.NamedAgg(column='trade_apply_amt', aggfunc='mean'),
        apply_amt_max = pd.NamedAgg(column='trade_apply_amt', aggfunc='max'),
        apply_amt_min = pd.NamedAgg(column='trade_apply_amt', aggfunc='min'),
    ).reset_index()
    logging.info('<<<< 计算历史时段特征 {0} >>>>'.format(his_features.shape))

    return his_features


def __calc_trade_lst(df_trade_tmp):

    lst_m_g = df_trade_tmp.groupby(['id', 'elapsed_month']).agg({
        'trade_cnt': [sum],
        'trade_apply_amt': [sum, np.mean, max, min],
    })
    values_col = ['_'.join(['u_p', x[0], x[1]]) for x in lst_m_g.columns]
    lst_m_g.columns = values_col
    lst_m_g = lst_m_g.reset_index()

    lst_features = pd.pivot_table(lst_m_g, index=['id'], columns='elapsed_month', values=values_col, fill_value=0)
    vars_col = ['_'.join([x[0], 'l'+str(x[1])+'m']) for x in lst_features.columns]
    lst_features.columns = vars_col
    lst_features = lst_features.reset_index()
    logging.info('<<<< 计算最近第d月特征 {0} >>>>'.format(lst_features.shape))

    return lst_features


def __calc_trade_pst(df_trade_tmp):

    months_list = list(set(df_trade_tmp['elapsed_month']))
    months_list.sort()

    pst_features = df_trade_tmp[['id']].drop_duplicates().reset_index(drop=True)
    for m in months_list:
        pst_m_g = df_trade_tmp.loc[df_trade_tmp['elapsed_month'] <= m].groupby(['id']).agg({
            'trade_cnt': [sum],
            'trade_apply_amt': [sum, np.mean, max, min],
        })
        vars_col = ['_'.join([x[0], x[1], 'p'+str(m)+'m']) for x in pst_m_g.columns]
        pst_m_g.columns = vars_col
        pst_m_g = pst_m_g.reset_index()
        pst_features = pd.merge(pst_features, pst_m_g, on=['id'], how='left')
    logging.info('<<<< 统计最近d月内特征 {0} >>>>'.format(pst_features.shape))

    return pst_features




