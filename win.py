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
from PyQt5 import QtCore, QtGui, QtWidgets
import sys,io
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import qdarkstyle
import multiprocessing
import okex.Account_api as Account
import okex.Funding_api as Funding
import okex.Market_api as Market
import okex.Public_api as Public
import okex.Trade_api as Trade
import okex.subAccount_api as SubAccount
import okex.status_api as Status
import math
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return json.JSONEncoder.default(self,obj)


class Datainfo:

    

    def eth_isbuy(minute,symbollist):

        for symbol in symbollist:

 
            t = time.time()

            #print (t)                       #ԭʼʱ������
            #print (int(t))                  #�뼶ʱ���
            #print (int(round(t * 1000)))    #���뼶ʱ���
            #print (int(round(t * 1000000))) #΢�뼶ʱ���
            tt = str((int(t * 1000)))
            ttt = str((int(round(t * 1000000))))
            
            
                


            #===��ȡclose����

        


            headers = {
            'authority': 'www.okex.com',
            'sec-ch-ua': '^\\^',
            'timeout': '10000',
            'x-cdn': 'https://static.okex.com',
            'devid': '7f1dea77-90cd-4746-a13f-a98bac4a333b',
            'accept-language': 'zh-CN',
            'sec-ch-ua-mobile': '?0',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36',
            'accept': 'application/json',
            'x-utc': '8',
            'app-type': 'web',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'cors',
            'sec-fetch-dest': 'empty',
            'referer': 'https://www.okex.com/markets/swap-info/'+symbol.lower()[:-5],
            'cookie': 'locale=zh_CN; _gcl_au=1.1.1849415495.'+str(tt)+'; _ga=GA1.2.1506507962.'+str(tt)+'; _gid=GA1.2.256681666.'+str(tt)+'; first_ref=https^%^3A^%^2F^%^2Fwww.okex.com^%^2Fcaptcha^%^3Fto^%^3DaHR0cHM6Ly93d3cub2tleC5jb20vbWFya2V0cy9zd2FwLWRhdGEvZXRoLXVzZA^%^3D^%^3D; _gat_UA-35324627-3=1; amp_56bf9d=gqC_GMDGl4q5Tk-BJhT-oP...1f711b989.1f711fv58.0.2.2',
            }

            params = (
            ('granularity', str(int(minute)*60)),
            ('size', '1000'),
            ('t', str(ttt)),
            )
            response = r.get('https://www.okex.com/v2/perpetual/pc/public/instruments/'+symbol+'/candles', headers=headers, params=params)

            if response.cookies.get_dict(): #����cookie��Ч 
                    s=r.session()
                    c = r.cookies.RequestsCookieJar()#����һ��cookie����
                    c.set('cookie-name', 'cookie-value')#����cookie��ֵ
                    s.cookies.update(c)#����s��cookie
                    s.get(url = 'https://www.okex.com/v2/perpetual/pc/public/instruments/'+symbol+'/candles?granularity=900&size=1000&t='+str(ttt))
            dw = pd.DataFrame(eval(json.dumps(response.json()))['data'])
            #print(df)
            dw.columns = ['timestamps','open','high','low','close','vol','p']
            datelist = []
            for timestamp in dw['timestamps']:
                datelist.append(timestamp.split('.000Z')[0].replace('T',' '))
            dw['timestamps'] = datelist
            dw['timestamps'] = pd.to_datetime(dw['timestamps'])+pd.to_timedelta('8 hours')
            #df['timestamps'] = df['timestamps'].apply(lambda x:time.mktime(time.strptime(str(x),'%Y-%m-%d %H:%M:%S')))
            #print(dw)
            dw['vol'] = list(map(float, dw['vol'].values))
            dw['close'] = list(map(float, dw['close'].values))
        
            dw.to_csv(f'./datas/okex/'+symbol+'/close.csv',index = False)
       
            dw = pd.read_csv(f'./datas/okex/'+symbol+'/close.csv')

            Datainfo.getfulldata(dw,symbol)

            learning = Datainfo.getnextdata(dw,symbol)


            #===�ж��Ƿ������������
        
            #df = pd.read_csv(f'./datas/okex/'+symbol+'/old_'+symbol+'.csv')

            if(dw['vol'].values[-1] and dw['p'].values[-1] and dw['vol'].values[-2] and dw['p'].values[-2] and learning):
                X1 = dw['close'].values[-1]/dw['vol'].values[-1]*dw['p'].values[-1]/dw['MA'].values[-1]*dw['obv'].values[-1]/dw['maobv'].values[-1]*dw['TRIX'].values[-1]*dw['MATRIX'].values[-1]*dw['close5'].values[-1]/dw['close135'].values[-1]*dw['macd'].values[-1]
                X2 = dw['close'].values[-2]/dw['vol'].values[-2]*dw['p'].values[-2]/dw['MA'].values[-2]*dw['obv'].values[-2]/dw['maobv'].values[-2]*dw['TRIX'].values[-2]*dw['MATRIX'].values[-2]*dw['close5'].values[-2]/dw['close135'].values[-2]*dw['macd'].values[-2]

                Y1 = dw['close'].values[-1]*float(dw['MATRIX'].values[-1])*float(dw['TRIX'].values[-1])
                Y2 = dw['close'].values[-2]*float(dw['MATRIX'].values[-2])*float(dw['TRIX'].values[-2])

                maxvalue = dw.iloc[-50:][(dw['macd'] == dw['macd'][-50:].max())]['close']
                minvalue = dw.iloc[-50:][(dw['macd'] == dw['macd'][-50:].min())]['close']
                value = maxvalue.values - minvalue.values
                value_618 = maxvalue.values - value * 0.618
                value_192 = maxvalue.values - value * 0.192

                if(not(X1 >5 and X2 < -3) and X1 >0 and X2 <0 and not(Y1 >0 and Y2 < 0) and dw['macd'].values[-1] > dw['macd'].values[-2] ):
                    api_key, secret_key, passphrase, flag = Datainfo.get_userinfo()
                    Datainfo.orderbuy(api_key, secret_key, passphrase, flag,symbol)
                elif(dw['close'].values[-1] > value_618 and dw['macd'].values[-1]>0 and dw['macd'].values[-2]<0  and dw['macd'].values[-1] > dw['macd'].values[-2] ):
                    api_key, secret_key, passphrase, flag = Datainfo.get_userinfo()
                    Datainfo.orderbuy(api_key, secret_key, passphrase, flag,symbol)
                elif(dw['macd'].values[-2] == dw['macd'][-40:].min() and dw['macd'].values[-2] < 0 and dw['macd'].values[-2] < dw['macd'].values[-1]):
                    api_key, secret_key, passphrase, flag = Datainfo.get_userinfo()
                    Datainfo.orderbuy(api_key, secret_key, passphrase, flag,symbol)

                       



    def getfulldata(df,symbol):


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
        df['obv'] = ta.OBV(df['close'].values,df['vol'].values)
        df['maobv'] = ta.MA(df['obv'].values, timeperiod=30, matype=0)

        df['TRIX'] = ta.TRIX(np.array(df['close'].values), timeperiod=14)
        df['MATRIX'] = ta.MA(df['TRIX'].values, timeperiod=30, matype=0)

        
        df.to_csv(f'./datas/okex/'+symbol+'/close.csv',index = False)

       

    #��ȡ�û�API��Ϣ
    def get_userinfo():

        with open('api.json', 'r', encoding='utf-8') as f:
            obj = json.loads(f.read())


        api_key = obj['api_key']
        secret_key = obj['secret_key']
        passphrase = obj['passphrase']

        # flag��ʵ����ģ���̵��л����� flag is the key parameter which can help you to change between demo and real trading.
        # flag = '1'  # ģ���� demo trading
        flag = '0'  # ʵ�� real trading

        return api_key,secret_key,passphrase,flag

    #������Ϣ
    def saveinfo(info):

        f_info = f'./datas/log/infodata.txt'
 

        with open(f_info,"a+",encoding='utf-8') as file:   #a :   д���ļ������ļ�����������ȴ�����д�룬�����Ḳ��ԭ�ļ�������׷�����ļ�ĩβ 
            file.write('\n'+str(info)+str(datetime.now()))

    
    #����������Ϣ
    def save_finalinfo(info):

        f_day = f'./datas/log/day_buy.txt'

        with open(f_day,"a+",encoding='utf-8') as file:   #a :   д���ļ������ļ�����������ȴ�����д�룬�����Ḳ��ԭ�ļ�������׷�����ļ�ĩβ 
            file.write('\n'+str(info)+'--->>>'+str(datetime.now()))


    
    #�����Զ��µ�
    def orderbuy(api_key, secret_key, passphrase, flag,symbol):


        # account api
        accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
        
        # ���óֲ�ģʽ  Set Position mode
        result = accountAPI.get_position_mode('long_short_mode')
        # ���øܸ˱���  Set Leverage
        result = accountAPI.set_leverage(instId=symbol.upper()+'-USD-SWAP', lever='100', mgnMode='cross')
        #Datainfo.saveinfo('����100����֤��ܸ���ϡ�����')
        # trade api
        tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
        # �����µ�  Place Multiple Orders
        # �����µ�  Place Multiple Orders
        result = tradeAPI.place_multiple_orders([
             {'instId': symbol, 'tdMode': 'cross', 'side': 'buy', 'ordType': 'market', 'sz': '1',
              'posSide': 'long',
              'clOrdId': 'a12344', 'tag': 'test1210'},
    

         ])
        print(result)

        #Datainfo.saveinfo('�µ���ϡ�����')

        lastprice = Datainfo.getlastprice(api_key, secret_key, passphrase, flag,symbol)

        #Datainfo.saveinfo('��ȡ���¼۸񡣡���'+str(lastprice))
        
        # ������֤��  Increase/Decrease margint
        result = accountAPI.Adjustment_margin(symbol, 'short', 'add', '5')
        #Datainfo.saveinfo('������֤����ϡ�����')

        # ����ί���µ�  Place Algo Order
        result = tradeAPI.place_algo_order(symbol, 'cross', 'sell', ordType='conditional',
                                            sz='1',posSide='long', tpTriggerPx=str(float(lastprice)*1.01), tpOrdPx=str(float(lastprice)*1.01))
        #Datainfo.saveinfo(str(datetime.now())+'����ֹӯ��ϡ�����'+str(float(lastprice)+50))


        sendtext = '����'+symbol+' -->> 1�ʣ��۸���'+str(lastprice)+'������ֹӯ��ϡ�����'+str(float(lastprice)*1.01)
        Datainfo.save_finalinfo(sendtext)
        SendDingding.sender(sendtext)



    #��ѯ���¼۸�
    def getlastprice(api_key, secret_key, passphrase, flag,symbol):

        # market api
        marketAPI = Market.MarketAPI(api_key, secret_key, passphrase, False, flag)

        # ��ȡ������Ʒ������Ϣ  Get Ticker
        result = marketAPI.get_ticker(symbol)
        print(eval(json.dumps(result['data'][0])))
        
        return eval(json.dumps(result['data'][0]))['last']
   
    #������ϴ
    def clean_data_df(df):
        # ���㵱ǰ��δ��1-day�ǵ���
        df.loc[:,'1d_close_future_pct'] = df['close'].shift(-1).pct_change(1)
        df.loc[:,'now_1d_direction'] = df['close'].pct_change(1)
        df.dropna(inplace=True)
        # ====1�������ǣ�0�����µ�
        df.loc[df['1d_close_future_pct'] > 0, 'future_1d_direction'] = 1
        df.loc[df['1d_close_future_pct'] <= 0, 'future_1d_direction'] = 0
        df = df[['now_1d_direction', 'future_1d_direction']]
        return df

    #�������ݱ�ǩ
    def split_train_and_test(df):
        # �������� X �ͱ�ǩ y
        y = df['future_1d_direction'].values
        X = df.drop('future_1d_direction', axis=1).values
        # ����ѵ�����Ͳ��Լ�
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.8, random_state=42)
        return X_train, X_test, y_train, y_test

    #svm_svcģ��
    def svm_svc(X_train, X_test, y_train, y_test):
        clf = svm.SVC(gamma='auto')
        clf.fit(X_train, y_train)
        new_prediction = clf.predict(X_test)
    #   print("Prediction: {}".format(new_prediction))
        return (clf.score(X_test, y_test))

    #������ SVM
    def main(df):
        #������ϴ
        df = Datainfo.clean_data_df(df)
        X_train, X_test, y_train, y_test =  Datainfo.split_train_and_test(df)
        svm_score = Datainfo.svm_svc(X_train, X_test, y_train, y_test)

    #��ȡ�¸�Ԥ����ֵ�ķ���
    def getnextdata(df,symbol):

        
        f_info = "\n��ʼ��ȡ�Ƿ�"+symbol+"�����ź� SVM�˹���������"
        print(f_info)
        #����������
        Datainfo.main(df)
        #��ȡcloseֵ
        for i in range(1, 21, 1):
            df['close - ' + str(i) + 'd'] = df['close'].shift(i)

        df_20d = df[[x for x in df.columns if 'close' in x]].iloc[20:]
        df_20d = df_20d.iloc[:,::-1]   # ת��������˳��

        #ѵ��ģ��
        clf = svm.SVR(kernel='linear')
        features_train = df_20d[:580]
        labels_train = df_20d['close'].shift(-1)[:580]     # �ع�����ı�ǩ����Ԥ��ľ��ǹɼۣ���һ������̼۾���ǰһ��ı�ǩ��
        features_test = df_20d[580:]
        labels_test = df_20d['close'].shift(-1)[580:]
        clf.fit(features_train, labels_train)     # ģ�͵�ѵ�����̣�

        predict = clf.predict(features_test)      # ������Լ������������ص��ǲ��Լ��ı�ǩ���ع�����ı�ǩ���ǹɼۣ�

        dft = pd.DataFrame(labels_test)
        dft['predict'] = predict     # ��ǰ��Ԥ��Ĳ��Լ��Ĺɼ۸���ӵ�DataFrame�У�
        dft = dft.rename(columns = {'close': 'Next Close', 'predict':'Predict Next Close'})

        current_close = df_20d[['close']].iloc[80:]
        next_open = df[['open']].iloc[82:].shift(-1)

        #��ȡdf1 df2��ֵ
        df1 = pd.merge(dft, current_close, left_index=True, right_index=True)

        df2 = pd.merge(df1, next_open, left_index=True, right_index=True)
        df2.columns = ['Next Close', 'Predicted Next Close', 'Current Close', 'Next Open']
        #��ͼ
        #df2['Signal'] = np.where(df2['Predicted Next Close'] > df2['Next Open'] ,1,0)

        #df2['PL'] =  np.where(df2['Signal'] == 1,(df2['Next Close'] - df2['Next Open'])/df2['Next Open'],0)

        #df2['Strategy'] = (df2['PL'].shift(1)+1).cumprod()
        #df2['return'] = (df2['Next Close'].pct_change()+1).cumprod()

        #df2[['Strategy','return']].dropna().plot(figsize=(10, 6))

        #��ȡԤ���¸������ֵ
        print(df2['Predicted Next Close'].tail(1).values[0] > df2['Current Close'].tail(1).values[0])
        print(df2.tail(5))
        return df2['Predicted Next Close'].tail(1).values[0] > df2['Current Close'].tail(1).values[0]
  
    class mywindows_multiprocessing ():

    
        #���к��� ����д
        def run():
            sch = Datainfo.windowsshow()

            #����6���̱�������
            p1 = multiprocessing.Process(target = sch.showwindows)
            p2 = multiprocessing.Process(target = sch.okex240M_buy)
            p3 = multiprocessing.Process(target = sch.okex60M_buy)

            #6�����̿�ʼ����
  
            p3.start()
            p2.start()
            p1.start()

            p3.join()
            p2.join()
            p1.join()
            
            

    class Ui_MainWindow(object):


        def setupUi(self, MainWindow):
            MainWindow.setObjectName("MainWindow")
            MainWindow.resize(1920, 1080)
            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")
            self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
            self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 1920, 951))
            self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
            self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
            self.verticalLayout.setContentsMargins(0, 0, 0, 0)
            self.verticalLayout.setObjectName("verticalLayout")
            self.dateTimeEdit = QtWidgets.QDateTimeEdit(self.verticalLayoutWidget)
            self.dateTimeEdit.setObjectName("dateTimeEdit")
            self.dateTimeEdit.setAlignment(Qt.AlignRight)
            #�����������ֵ����Сֵ���ڵ�ǰ���ڵĻ����ϣ���һ����ǰһ��
            self.dateTimeEdit.setMinimumDate(QDate.currentDate().addDays(-365))
            self.dateTimeEdit.setMaximumDate(QDate.currentDate().addDays(365))
            self.dateTimeEdit.setDisplayFormat("yyyy-MM-dd HH:mm:ss ddd a")
            #���������ؼ�������
            self.dateTimeEdit.setCalendarPopup(True)
            self.verticalLayout.addWidget(self.dateTimeEdit)

            self.textBrowserone = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
            self.textBrowserone.setObjectName("textBrowser")
            self.textBrowserone.setStyleSheet("background-color:#4e72b8;font-size:15px; font-family:'�п�';font-weight:bold;")
            self.textBrowserone.setAlignment(Qt.AlignCenter)
            self.textBrowserone.verticalScrollBar().setValue(self.textBrowserone.maximumHeight())
            self.verticalLayout.addWidget(self.textBrowserone)

            self.textBrowsertwo = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
            self.textBrowsertwo.setObjectName("textBrowser")
            self.textBrowsertwo.setStyleSheet("background-color:#9b95c9;font-size:15px; font-family:'�п�';font-weight:bold;")
            self.textBrowsertwo.setAlignment(Qt.AlignCenter)
            self.textBrowsertwo.verticalScrollBar().setValue(self.textBrowsertwo.maximumHeight())
            self.verticalLayout.addWidget(self.textBrowsertwo)
            

            MainWindow.setCentralWidget(self.centralwidget)
            self.menubar = QtWidgets.QMenuBar(MainWindow)
            self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 23))
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
       

            #oneAction = self.menu_3.addAction("����")
            #oneAction.triggered.connect(mywindows.run())

            self.menubar.setStyleSheet("font-size:20px; font-family:'LiSu'; margin:0;padding:0;text-align:center;")
            #������ʱ��
            self.Timer=QTimer()
            #��ʱ��ÿ1s����һ��
            self.Timer.start(60)
            #������ʱ������ͨ��  ע���������TimeUpdate���������Ƿ������صĵĽ�������Բ��ܴ����ţ�д��self.TimeUpdate()�ǲ��Ե�
            self.Timer.timeout.connect(self.TimeUpdate)


            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)

        



        def TimeUpdate(self):
        
            #'yyyy-MM-dd hh:mm:ss dddd' ���Ǹ�ʱ��ĸ�ʽ������yyyy�����꣬MM���£�dd���죬hh��Сʱ��mm�Ƿ��ӣ�ss���룬dddd������
            self.dateTimeEdit.setDateTime(QDateTime.currentDateTime()) 
          

            f_day = open(f'./datas/log/day_buy.txt',"r",encoding='utf-8')   #�����ļ�����
            day_buy = f_day.read()[-1600:]     #��txt�ļ����������ݶ��뵽�ַ���str��
            f_day.close()   #���ļ��ر�
            if(day_buy):
                self.textBrowsertwo.clear()
                self.textBrowsertwo.append(day_buy)

            f_info = open(f'./datas/log/infodata.txt',"r",encoding='utf-8')   #�����ļ�����
            infodata = f_info.read()[-1600:]     #��txt�ļ����������ݶ��뵽�ַ���str��
            f_info.close()   #���ļ��ر�
            if(infodata):
                self.textBrowserone.clear()
                self.textBrowserone.append(infodata)

            


        def retranslateUi(self, MainWindow):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
            self.menu.setTitle(_translate("MainWindow", "����"))
            self.menu_2.setTitle(_translate("MainWindow", "����"))
            self.menu_3.setTitle(_translate("MainWindow", "��ʼ"))
            self.action.setText(_translate("MainWindow", "ʲô�����ѧϰ"))

    class windowsshow():



        def showwindows(self):
            app = QtWidgets.QApplication(sys.argv) # ����һ��QApplication��Ҳ������Ҫ���������app
            MainWindow = QtWidgets.QMainWindow() # ����һ��QMainWindow������װ������Ҫ�ĸ���������ؼ�
            ui = Datainfo.Ui_MainWindow() # ui��Ui_MainWindow()���ʵ��������Ui_MainWindow��Ҫ�������ʵ���������objectname��Ĭ����MainWindow��
            ui.setupUi(MainWindow) # ִ�����е�setupUi�����������Ĳ����ǵڶ����д�����QMainWindow
            MainWindow.setWindowTitle("�˹����ܳ��򡪡���okex�Զ�����v3.0")
            # ���ز���
            dark_stylesheet=qdarkstyle.load_stylesheet_from_environment()# PyQtGraphdark_stylesheet=qdarkstyle.load_stylesheet_from_environment(is_pyqtgraph)
            app.setStyleSheet(dark_stylesheet)
            MainWindow.show() # ִ��QMainWindow��show()��������ʾ���QMainWindow
            Datainfo.textBrowserone = ui.textBrowserone
            Datainfo.textBrowsertwo = ui.textBrowsertwo

            ui.textBrowserone.append("��ӭ���� �˹����ܳ��򡪡�okex�Զ�����v3.0")
  
            sys.exit(app.exec_()) # ʹ��exit()���ߵ���رհ�ť�˳�QApp


        def okex5M_buy(self):


            self.getdatainfo('5')
            #scheduler = BlockingScheduler()
            #scheduler.add_job((self.getdatainfo), 'cron', args = ['5'], minute='*/5')
            #print(scheduler.get_jobs())
            #try:
                #scheduler.start()
            #except KeyboardInterrupt:
                #scheduler.shutdown()

        def okex15M_buy(self):


            #self.getdatainfo('15')
            scheduler = BlockingScheduler()
            scheduler.add_job((self.getdatainfo), 'cron', args = ['15'], minute='*/15')
            print(scheduler.get_jobs())
            try:
                scheduler.start()
            except KeyboardInterrupt:
                scheduler.shutdown()

        def okex3M_buy(self):


            #self.getdatainfo('15')
            scheduler = BlockingScheduler()
            scheduler.add_job((self.getdatainfo), 'cron', args = ['3'], minute='*/3')
            print(scheduler.get_jobs())
            try:
                scheduler.start()
            except KeyboardInterrupt:
                scheduler.shutdown()

        def okex60M_buy(self):


            #self.getdatainfo('15')
            scheduler = BlockingScheduler()
            scheduler.add_job((self.getdatainfo), 'cron', args = ['60'], hour='*/1')
            print(scheduler.get_jobs())
            try:
                scheduler.start()
            except KeyboardInterrupt:
                scheduler.shutdown()

        def okex240M_buy(self):


            #self.getdatainfo('15')
            scheduler = BlockingScheduler()
            scheduler.add_job((self.getdatainfo), 'cron', args = ['240'], hour='*/4')
            print(scheduler.get_jobs())
            try:
                scheduler.start()
            except KeyboardInterrupt:
                scheduler.shutdown()

        
        def getdatainfo(self,minute):

            
            
            t = time.time()

            #print (t)                       #ԭʼʱ������
            #print (int(t))                  #�뼶ʱ���
            #print (int(round(t * 1000)))    #���뼶ʱ���
            #print (int(round(t * 1000000))) #΢�뼶ʱ���
            tt = str((int(t * 1000)))
            ttt = str(int(round(t * 1000)))

            headers = {
                'authority': 'www.okex.com',
                'sec-ch-ua': '^\\^Chromium^\\^;v=^\\^94^\\^, ^\\^Google',
                'timeout': '10000',
                'x-cdn': 'https://static.okex.com',
                'devid': 'd2062109-476b-4fcf-95d2-b13156cb9915',
                'accept-language': 'zh-CN',
                'sec-ch-ua-mobile': '?0',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.54 Safari/537.36',
                'accept': 'application/json',
                'x-utc': '8',
                'app-type': 'web',
                'sec-ch-ua-platform': '^\\^Windows^\\^',
                'sec-fetch-site': 'same-origin',
                'sec-fetch-mode': 'cors',
                'sec-fetch-dest': 'empty',
                'referer': 'https://www.okex.com/markets/swap-coin-list',
                'cookie': 'locale=zh_CN; _gcl_au=1.1.307378181.1632526384; _ga=GA1.2.1775611624.'+str(tt)+'; _gid=GA1.2.1070759588.'+str(tt)+'; first_ref=https^%^3A^%^2F^%^2Fwww.okex.com^%^2Fmarkets^%^2Fspot-list; _gat_UA-35324627-3=1; amp_56bf9d=KkwRObhJezWtt8mdKRYolT...1fgdv5j6u.1fgdv60a6.1.1.2',
            }

            params = (
                ('t', str(tt)),
                ('instType', 'SWAP'),
            )

            response = r.get('https://www.okex.com/priapi/v5/public/open-interest', headers=headers, params=params)

            if response.cookies.get_dict(): #����cookie��Ч 
                s=r.session()
                c = r.cookies.RequestsCookieJar()#����һ��cookie����
                c.set('cookie-name', 'cookie-value')#����cookie��ֵ
                s.cookies.update(c)#����s��cookie
                s.get(url = 'https://www.okex.com/priapi/v5/public/open-interest?t='+str(tt)+'&instType=SWAP')

            df = pd.DataFrame(response.json()['data'])

            symbollist = list(df['instId'])

            symbols = []

            for i in symbollist:

                if(i[-8:-5]=='USD'):
                    symbols.append(i)

            p1=multiprocessing.Process(target = Datainfo.eth_isbuy,args=[minute,symbols[:5]]) 
            p2=multiprocessing.Process(target = Datainfo.eth_isbuy,args=[minute,symbols[5:10]])  
            p3=multiprocessing.Process(target = Datainfo.eth_isbuy,args=[minute,symbols[10:20]])  
            p4=multiprocessing.Process(target = Datainfo.eth_isbuy,args=[minute,symbols[20:30]])  
            p5=multiprocessing.Process(target = Datainfo.eth_isbuy,args=[minute,symbols[30:40]]) 
            p6=multiprocessing.Process(target = Datainfo.eth_isbuy,args=[minute,symbols[40:50]])  


 


            p1.start()
            p2.start()
            p3.start()
            p4.start()
            p5.start()
            p6.start()
  
   



            p1.join()
            p2.join()
            p3.join()
            p4.join()
            p5.join()
            p6.join()
   


   

           
#����������������
class SendDingding:
    def sender(sendtexts):
             
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
        


        params = {
            'sign' :sign,
    
            'timestamp' : timestamp
        }
        text_data ={
        "msgtype" :"text" ,
            "text":{
                "content": str(datetime.now())+'--->>>�������ػ��ߣ�Ҳ��һȺʱ�̶Կ�Σ�պͷ��Ŀ����� ��^_^     -->> '+sendtexts
            }
        }
        roboturl='https://oapi.dingtalk.com/robot/send?access_token=f8195c9e4ad6da4427d67e80dffed5d07ecaca1d1e79462fb5c0a9c6b12e90f2'
        r.post(roboturl, data=json.dumps(text_data),params=params, headers=headers )

if __name__ == '__main__':

  
    
    #����
    
  
    print('=============================================================================================')
    print('�������ػ��ߣ�Ҳ��һȺʱ�̶Կ�Σ�պͷ��Ŀ����� ��^_^')
    print('=============================================================================================')
    print('=============================================================================================')


    


    paths = []
    paths.append(f'./datas/okex/')
    paths.append(f'./datas/log/')

    t = time.time()

    #print (t)                       #ԭʼʱ������
    #print (int(t))                  #�뼶ʱ���
    #print (int(round(t * 1000)))    #���뼶ʱ���
    #print (int(round(t * 1000000))) #΢�뼶ʱ���
    tt = str((int(t * 1000)))
    ttt = str((int(round(t * 1000000))))

    headers = {
        'authority': 'www.okex.com',
        'sec-ch-ua': '^\\^Chromium^\\^;v=^\\^94^\\^, ^\\^Google',
        'timeout': '10000',
        'x-cdn': 'https://static.okex.com',
        'devid': 'd2062109-476b-4fcf-95d2-b13156cb9915',
        'accept-language': 'zh-CN',
        'sec-ch-ua-mobile': '?0',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.54 Safari/537.36',
        'accept': 'application/json',
        'x-utc': '8',
        'app-type': 'web',
        'sec-ch-ua-platform': '^\\^Windows^\\^',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
        'referer': 'https://www.okex.com/markets/swap-coin-list',
        'cookie': 'locale=zh_CN; _gcl_au=1.1.307378181.1632526384; _ga=GA1.2.1775611624.'+str(tt)+'; _gid=GA1.2.1070759588.'+str(tt)+'; first_ref=https^%^3A^%^2F^%^2Fwww.okex.com^%^2Fmarkets^%^2Fspot-list; _gat_UA-35324627-3=1; amp_56bf9d=KkwRObhJezWtt8mdKRYolT...1fgdv5j6u.1fgdv60a6.1.1.2',
    }

    params = (
        ('t', str(tt)),
        ('instType', 'SWAP'),
    )

    response = r.get('https://www.okex.com/priapi/v5/public/open-interest', headers=headers, params=params)

    if response.cookies.get_dict(): #����cookie��Ч 
        s=r.session()
        c = r.cookies.RequestsCookieJar()#����һ��cookie����
        c.set('cookie-name', 'cookie-value')#����cookie��ֵ
        s.cookies.update(c)#����s��cookie
        s.get(url = 'https://www.okex.com/priapi/v5/public/open-interest?t='+str(tt)+'&instType=SWAP')

    df = pd.DataFrame(response.json()['data'])

    symbollist = list(df['instId'])

    symbols = []

    for i in symbollist:

        if(i[-8:-5]=='USD'):
            symbols.append(i)
    #��txt�ļ����������ݶ��뵽�ַ���str��

    for symbol in symbols:
    
        paths.append(f'./datas/okex/'+symbol)
    

    for p in paths :
        try:
            os.makedirs(p,exist_ok=True)
        except:
            pass
    
    

    
    f_info = f'./datas/log/infodata.txt'
    f_day = f'./datas/log/day_buy.txt'

    #����ļ�����
    file = open(f'./datas/log/day_buy.txt', 'w',encoding='utf-8').close()
    file = open(f'./datas/log/infodata.txt', 'w',encoding='utf-8').close()


    with open(f_info,"a+",encoding='utf-8') as file:   #a :   д���ļ������ļ�����������ȴ�����д�룬�����Ḳ��ԭ�ļ�������׷�����ļ�ĩβ 
        file.write("��ʼ����okex API ��ȡ����==="+str(datetime.now()))

    

    mywindowsmultiprocessing = Datainfo.mywindows_multiprocessing
    mywindowsmultiprocessing.run()
