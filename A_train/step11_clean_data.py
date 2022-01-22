import os
import sys
import pickle

sys.path.append('.')
import config
from library import data_tool

# df_train
df_train = data_tool.load_df_train(data_dir=config.data_o_dir)
pickle.dump(df_train, open(os.path.join(config.data_p_dir, 'df_train.pkl'), 'wb'))

# df_test
df_test = data_tool.load_df_test(data_dir=config.data_o_dir)
pickle.dump(df_test, open(os.path.join(config.data_p_dir, 'df_test.pkl'), 'wb'))

# df_user_basic
df_user_basic = data_tool.load_df_user_basic(data_dir=config.data_o_dir)
pickle.dump(df_user_basic, open(os.path.join(config.data_p_dir, 'df_user_basic.pkl'), 'wb'))

# df_user_risk
df_user_risk = data_tool.load_df_user_risk(data_dir=config.data_o_dir)
pickle.dump(df_user_risk, open(os.path.join(config.data_p_dir, 'df_user_risk.pkl'), 'wb'))

# df_user_asset
df_user_asset = data_tool.load_df_user_asset(data_dir=config.data_o_dir)
pickle.dump(df_user_asset, open(os.path.join(config.data_p_dir, 'df_user_asset.pkl'), 'wb'))

# df_asset_info
# df_asset_info = data_tool.load_df_asset_info(data_dir=config.data_o_dir)
# pickle.dump(df_asset_info, open(os.path.join(config.data_p_dir, 'df_asset_info.pkl'), 'wb'))

# data_tool.load_df_asset_info_d1(data_dir=config.data_o_dir)


# df_trade
# df_trade = data_tool.load_df_trade(data_dir=config.data_o_dir)
# pickle.dump(df_trade, open(os.path.join(config.data_p_dir, 'df_trade.pkl'), 'wb'))

