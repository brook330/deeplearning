# -*- coding: gbk -*-

import pandas as pd
import talib as ta
import numpy as np
import time
df = pd.read_csv("c:/5min.csv")


#learning = Datainfo.getnextdata(df,'daoqiongsi')

def getfulldata(df,symbol,minute):


    #获取参数历史数据
        
    # MA - Moving average 移动平均线
    # 函数名：MA
    # 名称： 移动平均线
    # 简介：移动平均线，Moving Average，简称MA，原本的意思是移动平均，由于我们将其制作成线形，所以一般称之为移动平均线，简称均线。它是将某一段时间的收盘价之和除以该周期。 比如日线MA5指5天内的收盘价除以5 。
    # real = MA(close, timeperiod=30, matype=0)
    # 调用talib计算5\35\135日指数移动平均线的值
        

    df['close5'] = ta.EMA(np.array(df['close'].values.astype('float')), timeperiod=5)
    df['close35'] = ta.EMA(np.array(df['close'].values.astype('float')), timeperiod=35)
    df['close135'] = ta.EMA(np.array(df['close'].values.astype('float')), timeperiod=135)

    df["MACD_macd"],df["MACD_macdsignal"],df["MACD_macdhist"] = ta.MACD(df['close'].values.astype('float'), fastperiod=12, slowperiod=26, signalperiod=60)
    df['macd'] = 2*(df["MACD_macd"]-df["MACD_macdsignal"])

    df["MA"] = ta.MA(df['close'].values.astype('float'), timeperiod=30, matype=0)
    # EMA和MACD
    df['obv'] = ta.OBV(df['close'].values.astype('float'),df['vol'].values.astype('float'))
    df['maobv'] = ta.MA(df['obv'].values, timeperiod=30, matype=0)

    df['TRIX'] = ta.TRIX(np.array(df['close'].values.astype('float')), timeperiod=14)
    df['MATRIX'] = ta.MA(df['TRIX'].values, timeperiod=30, matype=0)            

getfulldata(df,'daoqiongsi','1')

for k in range(len(df['close'].values)):

   if(k>=300):
        bias=[]
        for i in range(k):
            if(i<=(k+34)):
                bias.append((df['close'].values[i]-df['close5'].values[i])/df['close5'].values[i])
        VAR1 = ta.EMA(np.array(bias), timeperiod=60)
        VAR2 = ta.EMA(np.array(VAR1), timeperiod=60)

        KONGPAN1= (VAR2[-1]-VAR2[-2])/VAR2[-2]
        KONGPAN2= (VAR2[-2]-VAR2[-3])/VAR2[-3]
        if(KONGPAN1>0 and KONGPAN1>KONGPAN2 and KONGPAN2<=0):


            X1 = df['close'].values[k-1]/df['vol'].values[k-1]/df['MA'].values[k-1]*df['obv'].values[k-1]/df['maobv'].values[k-1]*df['TRIX'].values[k-1]*df['MATRIX'].values[k-1]*df['close5'].values[k-1]/df['close135'].values[k-1]*df['macd'].values[k-1]
            X2 = df['close'].values[k-2]/df['vol'].values[k-2]/df['MA'].values[k-2]*df['obv'].values[k-2]/df['maobv'].values[k-2]*df['TRIX'].values[k-2]*df['MATRIX'].values[k-2]*df['close5'].values[k-2]/df['close135'].values[k-2]*df['macd'].values[k-2]

            Y1 = df['close'].values[k-1]*float(df['MATRIX'].values[k-1])*float(df['TRIX'].values[k-1])
            Y2 = df['close'].values[k-2]*float(df['MATRIX'].values[k-2])*float(df['TRIX'].values[k-2])

            

            if(not(X1 >5 and X2 < -3) and X1 >0 and X2 <0 and not(Y1 >0 and Y2 < 0)):

                print('-->>>买入-->>>'+str(df['date'].values[k-1])+'时间'+'--->>>买入价格--->>>'+str(df['close'].values[k-1])+'--->>>加油！！！--->>>')

                print(str(df['close'].values[k+1]))
                print(str(df['close'].values[k+2]))
                print(str(df['close'].values[k+3]))
                print(str(df['close'].values[k+4]))
                print(str(df['close'].values[k+5]))