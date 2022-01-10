import os

dirpath = os.path.dirname(os.path.dirname(__file__))

# data
data_o_dir = os.path.join(dirpath, 'data/A榜数据')
data_f_dir = os.path.join(dirpath, 'data/process')
data_v_dir = os.path.join(dirpath, 'data/variable')

# result
result_a_dir = os.path.join(dirpath, 'result/analysis')
result_p_dir = os.path.join(dirpath, 'result/predict')
result_m_dir = os.path.join(dirpath, 'result/model')
