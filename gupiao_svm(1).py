# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
from gm.api import *


# 可以直接提取数据，掘金终端需要打开，接口取数是通过网络请求的方式，效率一般，行情数据可通过subscribe订阅方式
# 设置token， 查看已有token ID,在用户-秘钥管理里获取
set_token('e080beaadbd377ab98cb64e3576e21d8e7d26ab6')
import threading
from concurrent.futures import ThreadPoolExecutor
import os,time
from apscheduler.schedulers.blocking import BlockingScheduler
import time,pyautogui,subprocess
import pandas as pd
from datetime import datetime,timedelta
import numpy as np
import talib as ta
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import hmac
import hashlib
import base64
import urllib.parse
import requests
import json
import shutil
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
import sys,io
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import Resource.resource
import qdarkstyle
import multiprocessing
import warnings
warnings.filterwarnings("ignore")


class Datainfo:


    
    #打印数据SHSE df_day
    def search_symbols_SHSE_data_day(symbol):
        
        
        data_day = history(symbol=symbol, frequency='1d', start_time=(datetime.now(
        )-timedelta(days=50000)).strftime('%Y-%m-%d'), end_time=(datetime.now()+timedelta(days=1)).strftime('%Y-%m-%d'),
                fields='bob,close,open,high,low,volume',adjust=ADJUST_PREV, 
                adjust_end_time=(datetime.now()+timedelta(days=1)).strftime('%Y-%m-%d'), df=True) 

        
        #单条日线股票数据大于等于500条
        if(data_day.shape[0]>200):
            data_day[['bob','close']].to_csv(f'./datas/day_SHSE/%s.csv'%symbol)
            #Datainfo.print_list.append("\n正在获取日线股票代码为："+symbol+"的数据")
            #print("\n正在获取日线股票代码为："+symbol+"的数据")
            #获取全部参数数据
            data_day = Datainfo.getfulldata(data_day)
            isbuy = Datainfo.buyinfo(data_day,symbol)
            if(isbuy):
                isok = Datainfo.getnextdata(data_day,symbol)
                if(isok):
                    f_info = f'./datas/day_SHSE/infodata.txt'
                    f_day = f'./datas/day_SHSE/day_buy.txt'
                
                    with open(f_info,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾
                        file.write('\n日线预测值 : %s'%(symbol))
                    with open(f_day,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾
                        file.write('\n日线预测值 : %s'%(symbol))
                    print('日线预测值 : %s'%(symbol))
            #保存所有data_day的数据
            #Datainfo.df_list_day.append(data_day[['bob','close']])

            

    #打印数据SHSE df_day
    def search_symbols_SZSE_data_day(symbol):
        
        
        data_day = history(symbol=symbol, frequency='1d', start_time=(datetime.now(
        )-timedelta(days=50000)).strftime('%Y-%m-%d'), end_time=(datetime.now()+timedelta(days=1)).strftime('%Y-%m-%d'),
                fields='bob,close,open,high,low,volume',adjust=ADJUST_PREV, 
                adjust_end_time=(datetime.now()+timedelta(days=1)).strftime('%Y-%m-%d'), df=True) 

        
        #单条日线股票数据大于等于500条
        if(data_day.shape[0]>200):
            
            data_day[['bob','close']].to_csv(f'./datas/day_SZSE/%s.csv'%symbol)
            #Datainfo.print_list.append("\n正在获取日线股票代码为："+symbol+"的数据")
            #print("\n正在获取日线股票代码为："+symbol+"的数据")
            f = f'./datas/day_SHSE/infodata.txt'
            #获取全部参数数据
            data_day = Datainfo.getfulldata(data_day)
            isbuy = Datainfo.buyinfo(data_day,symbol)
            if(isbuy):
                isok = Datainfo.getnextdata(data_day,symbol)
                if(isok):
                    f_info = f'./datas/day_SHSE/infodata.txt'
                    f_day = f'./datas/day_SHSE/day_buy.txt'
                
                    with open(f_info,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾
                        file.write('\n日线预测值 : %s'%(symbol))
                    with open(f_day,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾
                        file.write('\n日线预测值 : %s'%(symbol))
                    print('日线预测值 : %s'%(symbol))
            #保存所有data_day的数据
            #Datainfo.df_list_day.append(data_day[['bob','close']])

            
        


    #打印数据SHSE data_60M
    def search_symbols_SHSE_data_60M(symbol):
        
        
        data_60M = history(symbol=symbol, frequency='3600s', start_time=(datetime.now(
        )-timedelta(days=50000)).strftime('%Y-%m-%d'), end_time=(datetime.now()+timedelta(days=1)).strftime('%Y-%m-%d'),
                fields='bob,close,open,high,low,volume',adjust=ADJUST_PREV, 
                adjust_end_time=(datetime.now()+timedelta(days=1)).strftime('%Y-%m-%d'), df=True) 

        #单条股票数据大于等于500条
        if(data_60M.shape[0]>200):
            #Datainfo.print_list.append("\n正在获取60分钟股票代码为："+symbol+"的数据")
            print("\n正在获取60分钟股票代码为："+symbol+"的数据")
            data_60M[['bob','close']].to_csv(f'./datas/SHSE_60M/%s.csv'%symbol)
            #获取全部参数数据
            data_60M = Datainfo.getfulldata(data_60M)
            isbuy = Datainfo.buyinfo(data_60M)
            f = f'./datas/day_SHSE/infodata.txt'
            if(isbuy):
                isok = Datainfo.getnextdata(data_60M,symbol)
                if(isok):
                    f_info = f'./datas/day_SHSE/infodata.txt'
                    f_60M =f'./datas/day_SHSE/60M_buy.txt'
               
                    with open(f_info,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾
                        file.write('\n60分钟预测值 : %s'%(symbol))
                    with open(f_60M,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾
                        file.write('\n60分钟预测值 : %s'%(symbol))
                    print('60分钟预测值 : %s'%(symbol))
                
       

    #打印数据SZSE data_60M
    def search_symbols_SZSE_data_60M(symbol):

        data_60M = history(symbol=symbol, frequency='3600s', start_time=(datetime.now(
        )-timedelta(days=50000)).strftime('%Y-%m-%d'), end_time=(datetime.now()+timedelta(days=1)).strftime('%Y-%m-%d'),
                fields='bob,close,open,high,low,volume',adjust=ADJUST_PREV, 
                adjust_end_time=(datetime.now()+timedelta(days=1)).strftime('%Y-%m-%d'), df=True) 
        data_60M[['bob','close']].to_csv(f'./datas/SZSE_60M/%s.csv'%symbol)
        #单条股票数据大于等于500条
        if(data_60M.shape[0]>200):
            #Datainfo.print_list.append("\n正在获取60分钟股票代码为："+symbol+"的数据")
            print("\n正在获取60分钟股票代码为："+symbol+"的数据")

            #获取全部参数数据
            data_60M = Datainfo.getfulldata(data_60M)
            isbuy = Datainfo.buyinfo(data_60M)
            if(isbuy):
                isok = Datainfo.getnextdata(data_60M,symbol)
                if(isok):
                    f_info = f'./datas/day_SHSE/infodata.txt'
                    f_60M =f'./datas/day_SHSE/60M_buy.txt'
                
                    with open(f_info,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾
                        file.write('\n60分钟预测值 : %s'%(symbol))
                    with open(f_60M,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾
                        file.write('\n60分钟预测值 : %s'%(symbol))
                    print('60分钟预测值 : %s'%(symbol))
            

        
     #数据清洗
    def clean_data_df(df):
        # 计算当前、未来1-day涨跌幅
        df.loc[:,'1d_close_future_pct'] = df['close'].shift(-1).pct_change(1)
        df.loc[:,'now_1d_direction'] = df['close'].pct_change(1)
        df.dropna(inplace=True)
        # ====1代表上涨，0代表下跌
        df.loc[df['1d_close_future_pct'] > 0, 'future_1d_direction'] = 1
        df.loc[df['1d_close_future_pct'] <= 0, 'future_1d_direction'] = 0
        df = df[['now_1d_direction', 'future_1d_direction']]
        return df

    #增加数据标签
    def split_train_and_test(df):
        # 创建特征 X 和标签 y
        y = df['future_1d_direction'].values
        X = df.drop('future_1d_direction', axis=1).values
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.8, random_state=42)
        return X_train, X_test, y_train, y_test

    #svm_svc模型
    def svm_svc(X_train, X_test, y_train, y_test):
        clf = svm.SVC(gamma='auto')
        clf.fit(X_train, y_train)
        new_prediction = clf.predict(X_test)
    #   print("Prediction: {}".format(new_prediction))
        return (clf.score(X_test, y_test))

    #主函数 SVM
    def main(df):
        #数据清洗
        df = Datainfo.clean_data_df(df)
        X_train, X_test, y_train, y_test =  Datainfo.split_train_and_test(df)
        svm_score = Datainfo.svm_svc(X_train, X_test, y_train, y_test)

    #获取下个预期数值的方法
    def getnextdata(df,symbol):

        print("\n开始获取是否"+symbol+"买入信号 SVM人工智能运算")
        f_info = f'./datas/day_SHSE/infodata.txt'
        with open(f_info,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾
            file.write("\n开始获取是否"+symbol+"买入信号 SVM人工智能运算")
        #运行主函数
        Datainfo.main(df)
        #获取close值
        for i in range(1, 21, 1):
            df['close - ' + str(i) + 'd'] = df['close'].shift(i)

        df_20d = df[[x for x in df.columns if 'close' in x]].iloc[20:]
        df_20d = df_20d.iloc[:,::-1]   # 转换特征的顺序；

        #训练模型
        clf = svm.SVR(kernel='linear')
        features_train = df_20d[:200]
        labels_train = df_20d['close'].shift(-1)[:200]     # 回归问题的标签就是预测的就是股价，下一天的收盘价就是前一天的标签；
        features_test = df_20d[200:]
        labels_test = df_20d['close'].shift(-1)[200:]
        clf.fit(features_train, labels_train)     # 模型的训练过程；

        predict = clf.predict(features_test)      # 给你测试集的特征，返回的是测试集的标签，回归问题的标签就是股价；

        dft = pd.DataFrame(labels_test)
        dft['predict'] = predict     # 把前面预测的测试集的股价给添加到DataFrame中；
        dft = dft.rename(columns = {'close': 'Next Close', 'predict':'Predict Next Close'})

        current_close = df_20d[['close']].iloc[200:]
        next_open = df[['open']].iloc[220:].shift(-1)

        #获取df1 df2的值
        df1 = pd.merge(dft, current_close, left_index=True, right_index=True)

        df2 = pd.merge(df1, next_open, left_index=True, right_index=True)
        df2.columns = ['Next Close', 'Predicted Next Close', 'Current Close', 'Next Open']
        #画图
        df2['Signal'] = np.where(df2['Predicted Next Close'] > df2['Next Open'] ,1,0)

        df2['PL'] =  np.where(df2['Signal'] == 1,(df2['Next Close'] - df2['Next Open'])/df2['Next Open'],0)

        #df2['Strategy'] = (df2['PL'].shift(1)+1).cumprod()
        #df2['return'] = (df2['Next Close'].pct_change()+1).cumprod()

        #df2[['Strategy','return']].dropna().plot(figsize=(10, 6))

        #获取预期下个整点的值
        return df2['PL'][-2:-1].values[0]>0

    def getfulldata(df):
        #获取参数历史数据
        df['close5'] = ta.EMA(np.array(df['close'].values.astype('float')), timeperiod=5)
        df['close35'] = ta.EMA(np.array(df['close'].values.astype('float')), timeperiod=35)
        df['close135'] = ta.EMA(np.array(df['close'].values.astype('float')), timeperiod=135)

        df["MACD_macd"],df["MACD_macdsignal"],df["MACD_macdhist"] = ta.MACD(df['close'].values.astype('float'), fastperiod=12, slowperiod=26, signalperiod=60)
        df['macd'] = 2*(df["MACD_macd"]-df["MACD_macdsignal"])

        df["MA"] = ta.MA(df['close'].values.astype('float'), timeperiod=30, matype=0)
        # EMA和MACD
        df['obv'] = ta.OBV(df['close'].values.astype('float'),df['volume'].values.astype('float'))
        df['maobv'] = ta.MA(df['obv'].values, timeperiod=30, matype=0)

        df['TRIX'] = ta.TRIX(np.array(df['close'].values.astype('float')), timeperiod=14)
        df['MATRIX'] = ta.MA(df['TRIX'].values, timeperiod=30, matype=0)


        

        return df


        #判断并且发送买入信号
    def buyinfo(df,symbol):
        
        print("\n开始获取是否"+symbol+"买入信号 运算")
        f_info = f'./datas/day_SHSE/infodata.txt'
        with open(f_info,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾
            file.write("\n开始获取是否"+symbol+"买入信号 运算")
        if(dw['volume'].values[-1] and dw['volume'].values[-2]):
            X1 = dw['close'].values[-1]/dw['volume'].values[-1]/dw['MA'].values[-1]*dw['obv'].values[-1]/dw['maobv'].values[-1]*dw['TRIX'].values[-1]*dw['MATRIX'].values[-1]*dw['close5'].values[-1]/dw['close135'].values[-1]*dw['macd'].values[-1]
            X2 = dw['close'].values[-2]/dw['volume'].values[-2]/dw['MA'].values[-2]*dw['obv'].values[-2]/dw['maobv'].values[-2]*dw['TRIX'].values[-2]*dw['MATRIX'].values[-2]*dw['close5'].values[-2]/dw['close135'].values[-2]*dw['macd'].values[-2]

            Y1 = dw['close'].values[-1]*float(dw['MATRIX'].values[-1])*float(dw['TRIX'].values[-1])
            Y2 = dw['close'].values[-2]*float(dw['MATRIX'].values[-2])*float(dw['TRIX'].values[-2])

           

            if(dw['close'].values[-1]>dw['open'].values[-1]):
                print("\n计算完毕")
                return True
            else:
                print("\n计算完毕")
                return False 
        else:
            print("\n计算完毕")
            return False

    class mySearch_multiprocessing ():

     
        #运行函数 必须写
        def run(self):
            sch = Datainfo.RuntimeDatainfo()

            # 获取当天的日期
            today = datetime.now()
            # 获取上一个交易日
            last_day = get_previous_trading_date(exchange='SHSE', date=today)


            # 获取上证指数成份股
            SHSE_symbols = get_history_constituents(index='SHSE.000001', start_date=last_day, end_date=last_day)[0]['constituents'].keys() 

            # 获取深圳成指成份股
            SZSE_symbols = get_history_constituents(index='SZSE.399001', start_date=last_day, end_date=last_day)[0]['constituents'].keys()
            #计算股票数量
            SHSE_symbols_length = int(len(list(SHSE_symbols)))
            SHSE_symbols_length = int(len(list(SZSE_symbols)))


            #声明30进程
            p1 = multiprocessing.Process(target = sch.search_SHSE_symbols_day,args=[list(SHSE_symbols)[:int(SHSE_symbols_length*0.1)]])
            p2 = multiprocessing.Process(target = sch.search_SHSE_symbols_day,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.1):int(SHSE_symbols_length*0.2)]])
            p3 = multiprocessing.Process(target = sch.search_SHSE_symbols_day,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.2):int(SHSE_symbols_length*0.3)]])
            p4 = multiprocessing.Process(target = sch.search_SHSE_symbols_day,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.3):int(SHSE_symbols_length*0.4)]])
            p5 = multiprocessing.Process(target = sch.search_SHSE_symbols_day,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.4):int(SHSE_symbols_length*0.5)]])
            p6 = multiprocessing.Process(target = sch.search_SHSE_symbols_day,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.5):int(SHSE_symbols_length*0.6)]])
            p7 = multiprocessing.Process(target = sch.search_SHSE_symbols_day,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.6):int(SHSE_symbols_length*0.6)]])
            p8 = multiprocessing.Process(target = sch.search_SHSE_symbols_day,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.7):int(SHSE_symbols_length*0.7)]])
            p9 = multiprocessing.Process(target = sch.search_SHSE_symbols_day,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.8):int(SHSE_symbols_length*0.9)]])
            p10 = multiprocessing.Process(target = sch.search_SHSE_symbols_day,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.9):]])

            p11 = multiprocessing.Process(target = sch.search_SHSE_symbols_60M,args=[list(SHSE_symbols)[:int(SHSE_symbols_length*0.1)]])
            p12 = multiprocessing.Process(target = sch.search_SHSE_symbols_60M,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.1):int(SHSE_symbols_length*0.2)]])
            p13 = multiprocessing.Process(target = sch.search_SHSE_symbols_60M,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.2):int(SHSE_symbols_length*0.3)]])
            p14 = multiprocessing.Process(target = sch.search_SHSE_symbols_60M,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.3):int(SHSE_symbols_length*0.4)]])
            p15 = multiprocessing.Process(target = sch.search_SHSE_symbols_60M,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.4):int(SHSE_symbols_length*0.5)]])
            p16 = multiprocessing.Process(target = sch.search_SHSE_symbols_60M,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.5):int(SHSE_symbols_length*0.6)]])
            p17 = multiprocessing.Process(target = sch.search_SHSE_symbols_60M,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.6):int(SHSE_symbols_length*0.6)]])
            p18 = multiprocessing.Process(target = sch.search_SHSE_symbols_60M,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.7):int(SHSE_symbols_length*0.7)]])
            p19 = multiprocessing.Process(target = sch.search_SHSE_symbols_60M,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.8):int(SHSE_symbols_length*0.9)]])
            p20 = multiprocessing.Process(target = sch.search_SHSE_symbols_60M,args=[list(SHSE_symbols)[int(SHSE_symbols_length*0.9):]])


            p21 = multiprocessing.Process(target = sch.search_SZSE_symbols_day,args=[list(SZSE_symbols)[:int(SHSE_symbols_length*0.2)]])
            p22 = multiprocessing.Process(target = sch.search_SZSE_symbols_day,args=[list(SZSE_symbols)[int(SHSE_symbols_length*0.2):int(SHSE_symbols_length*0.4)]])
            p23 = multiprocessing.Process(target = sch.search_SZSE_symbols_day,args=[list(SZSE_symbols)[int(SHSE_symbols_length*0.4):int(SHSE_symbols_length*0.6)]])
            p24 = multiprocessing.Process(target = sch.search_SZSE_symbols_day,args=[list(SZSE_symbols)[int(SHSE_symbols_length*0.6):int(SHSE_symbols_length*0.8)]])
            p25 = multiprocessing.Process(target = sch.search_SZSE_symbols_day,args=[list(SZSE_symbols)[int(SHSE_symbols_length*0.8):]])

            p26 = multiprocessing.Process(target = sch.search_SZSE_symbols_60M,args=[list(SZSE_symbols)[:int(SHSE_symbols_length*0.2)]])
            p27 = multiprocessing.Process(target = sch.search_SZSE_symbols_60M,args=[list(SZSE_symbols)[int(SHSE_symbols_length*0.2):int(SHSE_symbols_length*0.4)]])
            p28 = multiprocessing.Process(target = sch.search_SZSE_symbols_60M,args=[list(SZSE_symbols)[int(SHSE_symbols_length*0.4):int(SHSE_symbols_length*0.6)]])
            p29 = multiprocessing.Process(target = sch.search_SZSE_symbols_60M,args=[list(SZSE_symbols)[int(SHSE_symbols_length*0.6):int(SHSE_symbols_length*0.8)]])
            p30 = multiprocessing.Process(target = sch.search_SZSE_symbols_60M,args=[list(SZSE_symbols)[int(SHSE_symbols_length*0.8):]])


            #30个线程开始运行
            p1.start()
            p2.start()
            p3.start()
            p4.start()
            p5.start()
            p6.start()
            p7.start()
            p8.start()
            p9.start()
            p10.start()
            p11.start()
            p12.start()
            p13.start()
            p14.start()
            p15.start()
            p16.start()
            p17.start()
            p18.start()
            p19.start()
            p20.start()
            p21.start()
            p22.start()
            p23.start()
            p24.start()
            p25.start()
            p26.start()
            p27.start()
            p28.start()
            p29.start()
            p30.start()

            #python中主线程等待子进程完成的实现(join())
            p1.join()
            p2.join()
            p3.join()
            p4.join()
            p5.join()
            p6.join()
            p7.join()
            p8.join()
            p9.join()
            p10.join()
            p11.join()
            p12.join()
            p13.join()
            p14.join()
            p15.join()
            p16.join()
            p17.join()
            p18.join()
            p19.join()
            p20.join()
            p21.join()
            p22.join()
            p23.join()
            p24.join()
            p25.join()
            p26.join()
            p27.join()
            p28.join()
            p29.join()
            p30.join()

  
    class mywindows_multiprocessing ():

    
        #运行函数 必须写
        def run():
            sch = Datainfo.windowsshow()

            #声明2线程保存数据
            p1 = multiprocessing.Process(target = sch.showwindows)
            p2 = multiprocessing.Process(target = sch.getdatainfo)
            

            #2个进程开始运行
            p2.start()
            p1.start()
            p2.join()
            p1.join()
            
            

    class Ui_MainWindow(object):


        def setupUi(self, MainWindow):
            MainWindow.setObjectName("MainWindow")
            MainWindow.resize(800, 600)
            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")
            self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
            self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 801, 551))
            self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
            self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
            self.verticalLayout.setContentsMargins(0, 0, 0, 0)
            self.verticalLayout.setObjectName("verticalLayout")
            self.dateTimeEdit = QtWidgets.QDateTimeEdit(self.verticalLayoutWidget)
            self.dateTimeEdit.setObjectName("dateTimeEdit")
            self.dateTimeEdit.setAlignment(Qt.AlignRight)
            #设置日期最大值与最小值，在当前日期的基础上，后一年与前一年
            self.dateTimeEdit.setMinimumDate(QDate.currentDate().addDays(-365))
            self.dateTimeEdit.setMaximumDate(QDate.currentDate().addDays(365))
            self.dateTimeEdit.setDisplayFormat("yyyy-MM-dd HH:mm:ss ddd a")
            #设置日历控件允许弹出
            self.dateTimeEdit.setCalendarPopup(True)
            self.verticalLayout.addWidget(self.dateTimeEdit)

            self.textBrowserone = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
            self.textBrowserone.setObjectName("textBrowser")
            self.textBrowserone.setStyleSheet("background-color:#4e72b8;font-size:20px; font-family:'行楷';font-weight:bold;")
            self.textBrowserone.setAlignment(Qt.AlignCenter)
            self.textBrowserone.verticalScrollBar().setValue(self.textBrowserone.maximumHeight())
            self.verticalLayout.addWidget(self.textBrowserone)

            self.textBrowsertwo = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
            self.textBrowsertwo.setObjectName("textBrowser")
            self.textBrowsertwo.setStyleSheet("background-color:#9b95c9;font-size:20px; font-family:'行楷';font-weight:bold;")
            self.textBrowsertwo.setAlignment(Qt.AlignCenter)
            self.textBrowsertwo.verticalScrollBar().setValue(self.textBrowsertwo.maximumHeight())
            self.verticalLayout.addWidget(self.textBrowsertwo)
            

            MainWindow.setCentralWidget(self.centralwidget)
            self.menubar = QtWidgets.QMenuBar(MainWindow)
            self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
            self.menubar.setObjectName("menubar")
            self.menu = QtWidgets.QMenu(self.menubar)
            self.menu.setObjectName("menu")
            self.menu_2 = QtWidgets.QMenu(self.menubar)
            self.menu_2.setObjectName("menu_2")
            self.menu_3 = QtWidgets.QMenu(self.menubar)
            self.menu_3.setObjectName("menu_3")
   
            MainWindow.setMenuBar(self.menubar)
            self.statusbar = QtWidgets.QStatusBar(MainWindow)
            self.statusbar.setObjectName("statusbar")
            MainWindow.setStatusBar(self.statusbar)
            self.action = QtWidgets.QAction(MainWindow)
            self.action.setObjectName("action")
            self.menu.addAction(self.action)
            self.menubar.addAction(self.menu_3.menuAction())
            self.menubar.addAction(self.menu.menuAction())
            self.menubar.addAction(self.menu_2.menuAction())
       

            #oneAction = self.menu_3.addAction("运行")
            #oneAction.triggered.connect(mywindows.run())

            self.menubar.setStyleSheet("font-size:20px; font-family:'LiSu'; margin:0;padding:0;text-align:center;")
            #创建定时器
            self.Timer=QTimer()
            #定时器每1s工作一次
            self.Timer.start(60)
            #建立定时器连接通道  注意这里调用TimeUpdate方法，不是方法返回的的结果，所以不能带括号，写成self.TimeUpdate()是不对的
            self.Timer.timeout.connect(self.TimeUpdate)


            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)

        



        def TimeUpdate(self):
        
            #'yyyy-MM-dd hh:mm:ss dddd' 这是个时间的格式，其中yyyy代表年，MM是月，dd是天，hh是小时，mm是分钟，ss是秒，dddd是星期
            self.dateTimeEdit.setDateTime(QDateTime.currentDateTime()) 
        
            
            f_60M= open(f'./datas/day_SHSE/60M_buy.txt',"r",encoding='utf-8')   #设置文件对象
            buy_60M = f_60M.read() #将txt文件的所有内容读入到字符串str中
            f_60M.close()   #将文件关闭
            if(buy_60M):
                self.textBrowsertwo.clear()
                self.textBrowsertwo.append(buy_60M)
                

            f_day = open(f'./datas/day_SHSE/day_buy.txt',"r",encoding='utf-8')   #设置文件对象
            day_buy = f_day.read()     #将txt文件的所有内容读入到字符串str中
            f_day.close()   #将文件关闭
            if(day_buy):
                self.textBrowsertwo.clear()
                self.textBrowsertwo.append(day_buy)

            f_info = open(f'./datas/day_SHSE/infodata.txt',"r",encoding='utf-8')   #设置文件对象
            infodata = f_info.read()     #将txt文件的所有内容读入到字符串str中
            f_info.close()   #将文件关闭
            if(infodata):
                self.textBrowserone.clear()
                self.textBrowserone.append(infodata)

            


        def retranslateUi(self, MainWindow):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
            self.menu.setTitle(_translate("MainWindow", "关于"))
            self.menu_2.setTitle(_translate("MainWindow", "帮助"))
            self.menu_3.setTitle(_translate("MainWindow", "开始"))
            self.action.setText(_translate("MainWindow", "什么是深度学习"))

    class windowsshow():



        def showwindows(self):
            app = QtWidgets.QApplication(sys.argv) # 创建一个QApplication，也就是你要开发的软件app
            MainWindow = QtWidgets.QMainWindow() # 创建一个QMainWindow，用来装载你需要的各种组件、控件
            ui = Datainfo.Ui_MainWindow() # ui是Ui_MainWindow()类的实例化对象，Ui_MainWindow需要根据你的实例化对象的objectname，默认是MainWindow。
            ui.setupUi(MainWindow) # 执行类中的setupUi方法，方法的参数是第二步中创建的QMainWindow
            MainWindow.setWindowTitle("人工智能程序——股票自动选股v3.0")
            pixmap = QPixmap (":/Images/account");
            MainWindow.setWindowIcon(QtGui.QIcon(pixmap))
            # 加载操作
            dark_stylesheet=qdarkstyle.load_stylesheet_from_environment()# PyQtGraphdark_stylesheet=qdarkstyle.load_stylesheet_from_environment(is_pyqtgraph)
            app.setStyleSheet(dark_stylesheet)
            MainWindow.show() # 执行QMainWindow的show()方法，显示这个QMainWindow
            Datainfo.textBrowserone = ui.textBrowserone
            Datainfo.textBrowsertwo = ui.textBrowsertwo

            ui.textBrowserone.append("欢迎运行 人工智能程序——股票自动选股v3.0")
  
            sys.exit(app.exec_()) # 使用exit()或者点击关闭按钮退出QApp

        def getdatainfo(self):

            #查询数据
            mysearch = Datainfo.mySearch_multiprocessing()
            mysearch.run()


    class RuntimeDatainfo:
    
            

        #上证指数获取数据方式
        def search_SHSE_symbols_day(self,SHSE_symbol_list):
            
            
            #开启线程池循环录入线程 上证指数 
            for symbol in SHSE_symbol_list:
                try:
                    Datainfo.search_symbols_SHSE_data_day(symbol)
                except:
                    time.sleep(0.3)
                    continue

        def search_SHSE_symbols_60M(self,SHSE_symbol_list):
            
            
            #开启线程池循环录入线程 上证指数 
            for symbol in SHSE_symbol_list:
                try:
                    Datainfo.search_symbols_SHSE_data_60M(symbol)
                except:
                    time.sleep(0.3)
                    continue

           
        #深圳成指获取数据方式
        def search_SZSE_symbols_day(self,SZSE_symbol_list):

            #开启线程池循环录入线程 深圳成指
            for symbol in SZSE_symbol_list:
                try:
                    Datainfo.search_symbols_SZSE_data_day(symbol)  
                except:
                    time.sleep(0.3)
                    continue

        def search_SZSE_symbols_60M(self,SZSE_symbol_list):

            #开启线程池循环录入线程 深圳成指
            for symbol in SZSE_symbol_list:
                try:
                    Datainfo.search_symbols_SZSE_data_60M(symbol)   
                except:
                    time.sleep(0.3)
                    continue

            
           

# ==================================================
# 最新数据获取出来然后录入到数据库，每隔60分钟
# ==================================================

#分配任务 按分钟分配 调用mvc获取正确的方法调用
class HSIdevelop:

    #60分钟
    def print_and_save_60M():


        if(datetime.now() == datetime.strptime(str(datetime.now().date())+' 09:30:00', '%Y-%m-%d %H:%M:%S') or datetime.now() ==datetime.strptime(str(datetime.now().date())+' 11:30:00', '%Y-%m-%d %H:%M:%S') or datetime.now() == datetime.strptime(str(datetime.now().date())+' 10:30:00', '%Y-%m-%d %H:%M:%S')):
            #启动掘金量化3
            #prs=subprocess.Popen(["C:/Users/何/AppData/Roaming/Hongshu Goldminer3/goldminer3.exe"])
            print('===========================10秒打开掘金3软件登录 这样才可以查询数据==========================================')
            time.sleep(3)
            print('=============================================================================================')
            print('我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫 ！^_^')
            print('=============================================================================================')
            print('=============================================================================================')
            print('=========开始执行早间服务======================')
            print('==============================60分钟开始=======================================================')
        
            datainfo = Datainfo()
            print("开始启动双线程，并且启动双线程池，30个分线程获取所有上证和深圳股票日数据.................................................")
            datainfo.mySearchThread.run() 

        elif(datetime.now() == datetime.strptime(str(datetime.now().date())+' 13:00:00', '%Y-%m-%d %H:%M:%S') or datetime.now() ==datetime.strptime(str(datetime.now().date())+' 14:00:00', '%Y-%m-%d %H:%M:%S') or datetime.now() == datetime.strptime(str(datetime.now().date())+' 15:00:00', '%Y-%m-%d %H:%M:%S')):
            #启动掘金量化3
            #prs=subprocess.Popen(["C:/Users/何/AppData/Roaming/Hongshu Goldminer3/goldminer3.exe"])
            print('===========================10秒打开掘金3软件登录 这样才可以查询数据==========================================')
            time.sleep(10)
            print('=============================================================================================')
            print('我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫 ！^_^')
            print('=============================================================================================')
            print('=============================================================================================')
            print('=========开始执行下午服务======================')
            print('==============================60分钟开始=======================================================')
        
            datainfo = Datainfo()
            print("开始启动双线程，并且启动双线程池，30个分线程获取所有上证和深圳股票日数据.................................................")
            datainfo.mySearchThread.run() 
        



        




           
#发钉钉的类先声明
class SendDingding:
    def sender(close,minutes,symbol):
             
        headers = {
        'Content-Type' : 'application/json'
        }
        timestamp = str(round(time.time()*1000))
        secret ="SEC050a3b2c9e5d8d0c777bbdd61270676a8bdad3608b36a086d70e95b712ad2db0"
        secret_enc = secret.encode('utf-8' )
        string_to_sign ='{}\n{}'.format(timestamp,secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        today = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        
        sendtexts = "本地时间： " +today+"，\n 预测买入 股票代码："+symbol+"，\n "+minutes+"分钟，\n现在的收盘价是："+str(close)+"\n"+ "，\n我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫！！！"

        params = {
            'sign' :sign,
    
            'timestamp' : timestamp
        }
        text_data ={
        "msgtype" :"text" ,
            "text":{
                "content": sendtexts
            }
        }
        roboturl='https://oapi.dingtalk.com/robot/send?access_token=f8195c9e4ad6da4427d67e80dffed5d07ecaca1d1e79462fb5c0a9c6b12e90f2'
        r=requests.post(roboturl, data=json.dumps(text_data),params=params, headers=headers )

if __name__ == '__main__':

  
    
    #启动掘金量化3
    
    print('===========================10秒打开掘金3软件登录 这样才可以查询数据==========================================')
    time.sleep(3)
    print('=============================================================================================')
    print('我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫 ！^_^')
    print('=============================================================================================')
    print('=============================================================================================')

    path1=f'./datas/day_SHSE/'
    path2=f'./datas/SHSE_60M/'

    #删除目录文件part  list
    if os.path.exists(path1):
        shutil.rmtree(path1)

    if os.path.exists(path2):
        shutil.rmtree(path2)
    time.sleep(3)

    #建立目录
        
    try:
        os.makedirs(path1,exist_ok=True)
        print("创建"+path1+"成功，或者目录已经存在")
    except:
        pass

    try:
        os.makedirs(path2,exist_ok=True)
        print("创建"+path2+"成功，或者目录已经存在")
    except:
        pass
    time.sleep(3)

    #清空文件内容
    file = open(f'./datas/day_SHSE/60M_buy.txt', 'w',encoding='utf-8').close()
    file = open(f'./datas/day_SHSE/day_buy.txt', 'w',encoding='utf-8').close()
    file = open(f'./datas/day_SHSE/infodata.txt', 'w',encoding='utf-8').close()

    time.sleep(3)
    mywindowsmultiprocessing = Datainfo.mywindows_multiprocessing
    mywindowsmultiprocessing.run()

    

    #保存数据

    #Datainfo.savedatalist(Datainfo.df_list_day)

    #mysavedataThread = Datainfo.mysavedataThread 
    #mysavedataThread.run()
        