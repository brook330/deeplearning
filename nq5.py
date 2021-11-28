# -*- coding: gbk -*-
import threading
import os,time,sys
from apscheduler.schedulers.blocking import BlockingScheduler
import datetime
import pandas as pd
from datetime import datetime,timedelta
import numpy as np
import talib as ta
import csv
import hmac
import talib as ta
import hashlib
import base64
import urllib.parse
import requests as  r
import json
import numpy as np
import time
import shutil
from pathlib import Path

def getfulldata(df):


    #��ȡ������ʷ����
        
    # MA - Moving average �ƶ�ƽ����
    # ��������MA
    # ���ƣ� �ƶ�ƽ����
    # ��飺�ƶ�ƽ���ߣ�Moving Average�����MA��ԭ������˼���ƶ�ƽ�����������ǽ������������Σ�����һ���֮Ϊ�ƶ�ƽ���ߣ���ƾ��ߡ����ǽ�ĳһ��ʱ������̼�֮�ͳ��Ը����ڡ� ��������MA5ָ5���ڵ����̼۳���5 ��
    # real = MA(close, timeperiod=30, matype=0)
    # ����talib����5\35\135��ָ���ƶ�ƽ���ߵ�ֵ
        

    df['close5'] = ta.EMA(np.array(df['close'].values), timeperiod=5)
    df['close35'] = ta.EMA(np.array(df['close'].values), timeperiod=35)
    df['close135'] = ta.EMA(np.array(df['close'].values), timeperiod=135)

    df["MACD_macd"],df["MACD_macdsignal"],df["MACD_macdhist"] = ta.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=60)
    df['macd'] = 2*(df["MACD_macd"]-df["MACD_macdsignal"])

    df["MA"] = ta.MA(df['close'].values, timeperiod=30, matype=0)
    # EMA��MACD
    df['obv'] = ta.OBV(df['close'].values,df['volume'].values)
    df['maobv'] = ta.MA(df['obv'].values, timeperiod=30, matype=0)

    df['TRIX'] = ta.TRIX(np.array(df['close'].values), timeperiod=14)
    df['MATRIX'] = ta.MA(df['TRIX'].values, timeperiod=30, matype=0)

df = pd.read_csv('������˹���ָ���ڻ�����-NQ-5min.csv')
df['volume']=df['volume'].astype(np.float64)
df['close']=df['close'].astype(np.float64)
df['open']=df['open'].astype(np.float64)
df['high']=df['high'].astype(np.float64)
df['low']=df['low'].astype(np.float64)
df.to_csv('������˹���ָ���ڻ�����-NQ-5min.csv',index=0)
print(df)
getfulldata(df)

success = 0
total=0
for num in range(len(df['close'].values)):

    if(num>200):
        dw = df[:num]
        bias=[]
        for i in range(len(dw['close'].values)):
            if(i<=(len(dw['close'].values)+34)):
                bias.append((dw['close'].values[i]-dw['close5'].values[i])/dw['close5'].values[i])
        VAR1 = ta.EMA(np.array(bias), timeperiod=60)
        VAR2 = ta.EMA(np.array(VAR1), timeperiod=60)

        KONGPAN1= (VAR2[-1]-VAR2[-2])/VAR2[-2]
        KONGPAN2= (VAR2[-2]-VAR2[-3])/VAR2[-3]
            
        X1 = dw['close'].values[-1]/dw['volume'].values[-1]/dw['MA'].values[-1]*dw['obv'].values[-1]/dw['maobv'].values[-1]*dw['TRIX'].values[-1]*dw['MATRIX'].values[-1]*dw['close5'].values[-1]/dw['close135'].values[-1]*dw['macd'].values[-1]
        X2 = dw['close'].values[-2]/dw['volume'].values[-2]/dw['MA'].values[-2]*dw['obv'].values[-2]/dw['maobv'].values[-2]*dw['TRIX'].values[-2]*dw['MATRIX'].values[-2]*dw['close5'].values[-2]/dw['close135'].values[-2]*dw['macd'].values[-2]

        Y1 = dw['close'].values[-1]*float(dw['MATRIX'].values[-1])*float(dw['TRIX'].values[-1])
        Y2 = dw['close'].values[-2]*float(dw['MATRIX'].values[-2])*float(dw['TRIX'].values[-2])
        #print(str(i)+','+str(dw['close'].values[-1])+','+str(KONGPAN1)+','+str(KONGPAN2)+','+str(X1)+','+str(X2)+','+str(Y1)+','+str(Y2)+','+str(dw['macd'].values[-1])+','+str(dw['macd'].values[-2]))
        #===�ж��Ƿ������������
        if(KONGPAN1>0 and KONGPAN2>0 and X1>0 and X2>0 and Y1>0 and Y2>0 and dw['macd'].values[-1]>0):           
                
                        
            if(KONGPAN1>KONGPAN2 and  X1>X2 and Y1>Y2 and dw['macd'].values[-1] > dw['macd'].values[-2]+3):  
                total+=1
                print(str(total)+','+str(dw['date'].values[-1])+','+str(dw['close'].values[-1])+','+str(dw['close'].values[-1]>dw['close'].values[-2]+30)+','+str(dw['close'].values[-1]>dw['close'].values[-3]+30)+','+str(dw['close'].values[-1]>dw['close'].values[-4]+30)+','+str(dw['close'].values[-1]>dw['close'].values[-5]+30)+','+str(dw['close'].values[-1]>dw['close'].values[-6]+30)+','+str(dw['close'].values[-1]>dw['close'].values[-7]+30)+','+str(dw['close'].values[-1]>dw['close'].values[-8]+30)+','+str(dw['close'].values[-1]>dw['close'].values[-9]+30))
                if(dw['close'].values[-1]>dw['close'].values[-2]+30 or dw['close'].values[-1]>dw['close'].values[-3]+30 or dw['close'].values[-1]>dw['close'].values[-4]+30 or dw['close'].values[-1]>dw['close'].values[-5]+30 or dw['close'].values[-1]>dw['close'].values[-6]+30 or dw['close'].values[-1]>dw['close'].values[-7]+30 or dw['close'].values[-1]>dw['close'].values[-8]+30 or dw['close'].values[-1]>dw['close'].values[-9]+30):
                    success+=1

if(success>0 and total>0):
            print('�ɹ���--->>>',success/total,'�ɹ���--->>>',success,'����--->>>',total)