import os

dirpath = os.path.dirname(os.path.dirname(__file__))

# data
data_o_dir = os.path.join(dirpath, 'data/A榜数据')
data_p_dir = os.path.join(dirpath, 'data/A榜数据/process')
data_f_dir = os.path.join(dirpath, 'data/A榜数据/features')


# result
result_o_dir = os.path.join(dirpath, 'result/A榜数据/origin')
result_p_dir = os.path.join(dirpath, 'result/A榜数据/process')
result_f_dir = os.path.join(dirpath, 'result/A榜数据/feature')
result_m_dir = os.path.join(dirpath, 'result/A榜数据/model')

# modelparm
random_seed = 2022
ignore_cols = ['id', 'core_cust_id', 'prod_code', 'prod_type', 'deal_date', 'y']