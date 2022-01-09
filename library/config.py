import os

dirpath = os.path.dirname(os.path.dirname(__file__))

# data
data_o_dir = os.path.join(dirpath, 'data/A榜数据')
data_f_dir = os.path.join(dirpath, 'data/process')

# result
result_a_dir = os.path.join(dirpath, 'result/analysis')
result_p_dir = os.path.join(dirpath, 'result/predict')

# model
model_dir = os.path.join(dirpath, 'model')
