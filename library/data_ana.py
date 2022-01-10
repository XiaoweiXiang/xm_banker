#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
本文档主要用于即席数据分析，基于python读取csv数据，查看数据样例
'''

import pandas as pd
from pandasql import *
import re

class DataAna(object):
    """
    提供各类数据分析工具
    load_data：加载数据集
    sql_query：提供数据SQL查询接口
    """

    def __init__(self,filename,path):
        self.filename = filename
        self.path = path

    def load_data(self):
        """
        数据加载函数，提供文件名及文件地址，返回df数据集       

       Parameters
        ----------
        self.filename : 字符串
         需要加载的文件名
        self.path :字符串，提供'A榜数据'的路径
         数据集路径        

       Returns
        -------
        DataFrame
         返回一个df格式数据集        
       """
        if self.filename in ['x_test','x_train','y_train']:
            path = self.path + '主表数据/'
        else :
            path = self.path + '其他数据表/'
        data = pd.read_csv(path+self.filename+'.csv')     

        return(data)


    def sql_query(self,df,sql):
        """
        数据加载函数，提供文件名及文件地址，返回df数据集       

       Parameters
        ----------
        df:DataFrame
         需要查询的数据集
        sql : 字符串
         sql 需要查询的代码      

       Returns
        -------
        DataFrame
         返回一个df格式的查询结果        
        """
        sql = sql.replace('\n','').lower()
        self.filename = re.findall("from(.*?)l", sql+'lllll')[0].replace(' ','')
        sql = sql.replace(self.filename,'df')
        #执行SQL
        result = sqldf(sql)

        return(result)



if __name__ == '__main__':

    path = '/Users/jiangdehao/Desktop/project/kaggle/data/A榜数据/'
    filename='j'
    
    myDataAna = DataAna(filename,path)
    sql='''
    select prod_code
    from df
    limit 10
    '''
    df = myDataAna.load_data()
    df = myDataAna.sql_query(df,sql)
    print(df)
