import os
import sys
import pickle
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

sys.path.append('.')
import config
from library import feature_tool, data_feature


sample_name = input('请输入样本集名称（train 或 test）：')
df_sample = pickle.load(open(os.path.join(config.data_p_dir, 'df_{0}.pkl'.format(sample_name)), 'rb'))
logging.info('读取{0}样本 df_sample={1} >>'.format(sample_name, df_sample.shape))

# 用户/客户信息特征
feature_u_info = feature_tool.proc_u_info(df_sample=df_sample, data_dir=config.data_p_dir)
pickle.dump(feature_u_info, open(os.path.join(config.data_f_dir, sample_name, 'feature_u_info.pkl'), 'wb'))

# 用户/风险评估行为信息特征
feature_u_risk = feature_tool.proc_u_risk(df_sample=df_sample, data_dir=config.data_p_dir, max_delta=6)
pickle.dump(feature_u_risk, open(os.path.join(config.data_f_dir, sample_name, 'feature_u_risk.pkl'), 'wb'))

# 用户/资产信息特征
feature_u_asset = feature_tool.proc_u_asset(df_sample=df_sample, data_dir=config.data_p_dir)
pickle.dump(feature_u_asset, open(os.path.join(config.data_f_dir, sample_name, 'feature_u_asset.pkl'), 'wb'))

# 理财产品/产品信息特征
feature_p_info = feature_tool.proc_p_info(df_sample=df_sample, data_dir=config.data_p_dir)
pickle.dump(feature_p_info, open(os.path.join(config.data_f_dir, sample_name, 'feature_p_info.pkl'), 'wb'))

# 用户x理财产品交易流水特征
feature_u_trade = feature_tool.proc_u_trade(df_sample=df_sample, data_dir=config.data_p_dir)
pickle.dump(feature_u_trade, open(os.path.join(config.data_f_dir, sample_name, 'feature_u_trade.pkl'), 'wb'))

feature_u_p_trade = feature_tool.proc_u_p_trade(df_sample=df_sample, data_dir=config.data_p_dir)
pickle.dump(feature_u_p_trade, open(os.path.join(config.data_f_dir, sample_name, 'feature_u_p_trade.pkl'), 'wb'))

# 用户信息特征
# path = config.data_o_dir+'/'
# train_ind = sample_name == 'train'
# myDataFeature = data_feature.DataFeature(path, train=train_ind)
# myDataFeature.basci_info_feature()
# myDataFeature.app_info_feature(time_dff_list=[3,7,15,30,90,180,360])
