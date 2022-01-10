#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
本文档主要用于特征衍生，输入原始数据路径，输出csv特征数据
'''
import sys
from data_ana import *
import pandas as pd
import numpy as np

class DataFeature(object):
    """
    提供各类数据衍生工具
    basci_info_feature：合并客户信息表、客户风险表、资产信息表
    """

    def __init__(self,path,train=True):
        self.path = path
        self.train = train

    def train_or_test(self):
        if self.train==True:
            filename='x_train'
        else:
            filename='x_test'
        return(filename)        

    def get_head_table(self):
        filename=self.train_or_test()
        self.myDataAna = DataAna(filename=filename,path=self.path) 
        result = self.myDataAna.load_data()
        result.columns = ['id','core_cust_id','prod_code','prod_type','trade_date']
        return(result)     

    def basci_info_feature(self,name_list=['d','e','f']):
        """
        合并客户信息表、客户风险表、资产信息表，返回df数据集并存储到features目录       

       Parameters
        ----------
        name_list : 文件名list，默认是['d','e','f']
         需要加载的文件名      

       Returns
        -------
        DataFrame
         返回一个df格式数据集，存储在path的features位置        
         """
        result = self.get_head_table()
        for i in name_list:
            self.myDataAna.filename=i
            df = self.myDataAna.load_data()
            result = result.merge(self.myDataAna.load_data(),on='core_cust_id')
        
        #去重
        result['f22'] = pd.to_datetime(result['f22'],format='%Y%m%d')
        result['dff'] = (abs(pd.to_datetime(result['trade_date'])-result['f22'])/pd.Timedelta(1, 'D')).fillna(0).astype(int)
        result['rank'] = result['dff'].groupby(result['id']).rank()
        result = result[result['rank']==1]
        result = result[[ i for i in result.columns if i not in ['core_cust_id','prod_code','prod_type','trade_date','e2','f1','f22','dff','rank']]]
        result = result.fillna(-999)
        result.to_csv(path+'/features/'+filename[2:100]+'_feature_basci.csv',index=False)
        print("===================>basic info feature merge success")


    def prod_type(self):
        """
        按照产品code类型，返回A、B、C、D标签

       Parameters
        ----------
        None      

       Returns
        -------
        DataFrame
         返回一个df格式数据集        
         """
        self.myDataAna = DataAna(filename=None,path=self.path)
        name = ['A','B','C','D']
        ls = ['g','h','i','j']
        result = pd.DataFrame()

        for i in ls:
            self.myDataAna.filename=i
            df = self.myDataAna.load_data()
            df['type'] = name[ls.index(i)]
            result = result.append(df[['prod_code','type']])
        return(result)

    def app_info_df(self):
        """
        拼接训练样本及APP信息，返回不同样本在交易日之前的APP点击明细信息(df)

       Parameters
        ----------
        None      

       Returns
        -------
        DataFrame
         返回一个df格式数据集        
         """
        self.myDataAna = DataAna(filename='r',path=self.path)
        app_df = self.myDataAna.load_data()
        app_info = self.prod_type()
        app_df = app_df.merge(app_info,on='prod_code',how='left')
        app_df['type'] = app_df['type'].fillna('other')
        
        head_table = self.get_head_table()
        df = head_table.merge(app_df[['core_cust_id','r3','type','r5']],on='core_cust_id')
        df[['trade_date']] = pd.to_datetime(df['trade_date'])
        df[['r5']] = pd.to_datetime(df['r5'])
        df = df[df['trade_date']>=df['r5']]
        return(df)


    def app_info_feature(self,time_dff_list=[1]):
        df = self.app_info_df()
        result = df[['id']].drop_duplicates()
        for time_dff in time_dff_list:
            print('==================>begin run time_dff:{}'.format(time_dff))
            mydf = df[df['trade_date'] - pd.Timedelta(days=time_dff)>=df['r5']]
            mydf['all'] = 1
            mydf['r3_1'] = np.where(mydf['r3']==1,1,0)
            mydf['type_a'] = np.where(mydf['type']=='A',1,0)
            mydf['type_b'] = np.where(mydf['type']=='B',1,0)
            mydf['type_c'] = np.where(mydf['type']=='C',1,0)
            mydf['type_d'] = np.where(mydf['type']=='D',1,0)
            mydf['hour'] = mydf['r5'].astype(str).str[11:13].astype(int)
            mydf['r5_0_6'] = np.where( mydf['hour']<=5,1,0)
            mydf['r5_6_11'] = np.where((mydf['hour']>=6) & (mydf['hour']<=11),1,0)
            mydf['r5_12_17'] = np.where((mydf['hour']>=12) & (mydf['hour']<=17),1,0)
            mydf['r5_18_23'] = np.where((mydf['hour']>=18) & (mydf['hour']<=23),1,0)
            result_new = pd.DataFrame(mydf.groupby('id')[['all','r3_1','type_a','type_b','type_c','type_d','r5_0_6','r5_6_11','r5_12_17','r5_18_23']].sum())
            result_new=result_new.reset_index()
            result_new.columns = ['id']+['feature_app_dtedff_'+str(time_dff)+'_'+i for i in result_new.columns if i != 'id']
            result = result.merge(result_new,on='id',how='left')
        
        result = result.fillna(-999)
        filename=self.train_or_test()
        result.to_csv(path+'/features/'+filename[2:100]+'_feature_app.csv',index=False)
        print("===================>app info feature merge success")


if __name__ == '__main__':
    path = '/Users/jiangdehao/Desktop/project/kaggle/data/A榜数据/'
    myDataFeature = DataFeature(path)
    # myDataFeature.basci_info_feature()
    myDataFeature.app_info_feature(time_dff_list=[3,7,15,30,90,180,360])
    # data = pd.read_csv('/Users/jiangdehao/Desktop/project/kaggle/data/A榜数据/features/train_feature_basci.csv')


