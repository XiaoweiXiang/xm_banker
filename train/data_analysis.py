import os 
import numpy as np
import pandas as pd
import config, load_o_data


df_train = load_o_data.load_df_train()
print(df_train.groupby(['core_cust_id']).agg({'prod_type': len, 'y': np.sum}))