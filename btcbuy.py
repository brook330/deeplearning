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
import warnings
warnings.filterwarnings('ignore')


class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return json.JSONEncoder.default(self,obj)


class Datainfo:

    

    def eth_isbuy(minute,symbol):

        result = '不买卖'
        ones = ''

        for i in range(10000):

            try:
                t = time.time()

                #print (t)                       #原始时间数据
                #print (int(t))                  #秒级时间戳
                #print (int(round(t * 1000)))    #毫秒级时间戳
                #print (int(round(t * 1000000))) #微秒级时间戳
                tt = str((int(t * 1000)))
                ttt = str(int(round(t * 1000)))

                #=====获取vol数据
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
                    'referer': 'https://www.okex.com/markets/swap-data/'+symbol+'-usd',
                    'cookie': '_gcl_au=1.1.1849415495.'+str(tt)+'; _ga=GA1.2.1506507962.'+str(tt)+'; first_ref=https^%^3A^%^2F^%^2Fwww.okex.com^%^2Fcaptcha^%^3Fto^%^3DaHR0cHM6Ly93d3cub2tleC5jb20vbWFya2V0cy9zd2FwLWRhdGEvZXRoLXVzZA^%^3D^%^3D; locale=zh_CN; _gid=GA1.2.802198982.'+str(tt)+'; amp_56bf9d=gqC_GMDGl4q5Tk-BJhT-oP...1f8fiso4n.1f8fiu841.1.2.3',
                }

                params = (
                    ('t', str(ttt)),
                    ('unitType', '0'),
                )

                response = r.get('https://www.okex.com/v3/futures/pc/market/takerTradeVolume/'+symbol.upper()+'', headers=headers, params=params)

                if response.cookies.get_dict(): #保持cookie有效 
                        s=r.session()
                        c = r.cookies.RequestsCookieJar()#定义一个cookie对象
                        c.set('cookie-name', 'cookie-value')#增加cookie的值
                        s.cookies.update(c)#更新s的cookie
                        s.get(url = 'https://www.okex.com/v3/futures/pc/market/takerTradeVolume/'+symbol.upper()+'?t='+str(ttt)+'&unitType=0')
                df = pd.DataFrame(response.json()['data'])

                df['timestamps'] = list(map(float, df['timestamps'].values))
                df['timestamps'] = pd.to_datetime(df['timestamps'],unit='ms')+pd.to_timedelta('8 hours')

                #列倒序内容排列
                df = df.iloc[:,::-1]




                df.to_csv(f'./datas/okex/'+symbol+'/old_'+symbol+'.csv',index=False)


                #===获取close数据

        


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
                'referer': 'https://www.okex.com/markets/swap-info/'+symbol+'-usd',
                'cookie': 'locale=zh_CN; _gcl_au=1.1.1849415495.'+str(tt)+'; _ga=GA1.2.1506507962.'+str(tt)+'; _gid=GA1.2.256681666.'+str(tt)+'; first_ref=https^%^3A^%^2F^%^2Fwww.okex.com^%^2Fcaptcha^%^3Fto^%^3DaHR0cHM6Ly93d3cub2tleC5jb20vbWFya2V0cy9zd2FwLWRhdGEvZXRoLXVzZA^%^3D^%^3D; _gat_UA-35324627-3=1; amp_56bf9d=gqC_GMDGl4q5Tk-BJhT-oP...1f711b989.1f711fv58.0.2.2',
                }

                params = (
                ('granularity', str(int(minute)*60)),
                ('size', '1000'),
                ('t', str(ttt)),
                )
                response = r.get('https://www.okex.com/v2/perpetual/pc/public/instruments/'+symbol.upper()+'-USD-SWAP/candles', headers=headers, params=params)

                if response.cookies.get_dict(): #保持cookie有效 
                        s=r.session()
                        c = r.cookies.RequestsCookieJar()#定义一个cookie对象
                        c.set('cookie-name', 'cookie-value')#增加cookie的值
                        s.cookies.update(c)#更新s的cookie
                        s.get(url = 'https://www.okex.com/v2/perpetual/pc/public/instruments/'+symbol.upper()+'-USD-SWAP/candles?granularity=900&size=1000&t='+str(ttt))
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
                time.sleep(int(minute))
                dw = pd.read_csv(f'./datas/okex/'+symbol+'/close.csv')
                time.sleep(int(minute))

                Datainfo.getfulldata(dw,symbol)


                #===判断是否买入或者卖出
        
                #df = pd.read_csv(f'./datas/okex/'+symbol+'/old_'+symbol+'.csv')

                if(dw['vol'].values[-1] and dw['p'].values[-1] and dw['vol'].values[-2] and dw['p'].values[-2]):
                    X1 = dw['close'].values[-1]/dw['vol'].values[-1]*dw['p'].values[-1]/dw['MA'].values[-1]*dw['obv'].values[-1]/dw['maobv'].values[-1]*dw['TRIX'].values[-1]*dw['MATRIX'].values[-1]*dw['close5'].values[-1]/dw['close135'].values[-1]*dw['macd'].values[-1]
                    X2 = dw['close'].values[-2]/dw['vol'].values[-2]*dw['p'].values[-2]/dw['MA'].values[-2]*dw['obv'].values[-2]/dw['maobv'].values[-2]*dw['TRIX'].values[-2]*dw['MATRIX'].values[-2]*dw['close5'].values[-2]/dw['close135'].values[-2]*dw['macd'].values[-2]

                    Y1 = dw['close'].values[-1]*float(dw['MATRIX'].values[-1])*float(dw['TRIX'].values[-1])
                    Y2 = dw['close'].values[-2]*float(dw['MATRIX'].values[-2])*float(dw['TRIX'].values[-2])


                    if( X1 >0 and X2 <0 and not(Y1 >0 and Y2 < 0)  and dw['macd'].values[-1] > dw['macd'].values[-2]):
                        result = '买入'
                        ones = '满足条件1'
                    
                    maxvalue = dw.iloc[-30:][(dw['macd'] == dw['macd'][-30:].max())]['close']
                    minvalue = dw.iloc[-30:][(dw['macd'] == dw['macd'][-30:].min())]['close']
                    value = maxvalue.values - minvalue.values
                    value_618 = maxvalue.values - value * 0.618
                    value_192 = maxvalue.values - value * 0.192

                    buyVolumes = df['buyVolumes'].tail(20).values
                    sellVolumes = df['sellVolumes'].tail(20).values

                    if(dw['close'].values[-1] > value_618 and dw['close'].values[-1] < value_192  and dw['macd'].values[-1] > dw['macd'].values[-2]+5 and (sum(buyVolumes)/len(buyVolumes)) / (sum(sellVolumes)/len(sellVolumes)) > 1.01):
                        result = '买入'
                        ones = '黄金分割公式'
                    if(dw['macd'].values[-2] == dw['macd'][-30:].min() and dw['macd'].values[-2] < 0 and dw['macd'].values[-2] < dw['macd'].values[-1]):
                        result = '买入'
                        ones = '最小值买入公式'
                    if(dw['macd'].values[-1]>dw['macd'].values[-2]+8 and dw['macd'].values[-2] > dw['macd'].values[-3] and dw['close'].values[-1] > dw['open'].values[-1] and (sum(buyVolumes)/len(buyVolumes)) / (sum(sellVolumes)/len(sellVolumes)) > 1.01 and dw['close'].values[-1] >value_192):
                        result = '买入'
                        ones = '神圣计划公式'

                    #哥要上涨公式
                    var1 = ((dw['close'].values+dw['open'].values+dw['high'].values+dw['low'].values)/4)[-1:]
                    var2 = ta.SMA(abs(np.array(dw['low'].values[-800:]-var1[-800:])),13)
                    var3 = ta.EMA(var2,10)
                    var3 = var3[~np.isnan(var3)]
                    var4 = dw['low'][-33:].min() 
                    var5 = 0
                    if(dw['low'].values[-1] < var4):
                        var5 = ta.EMA(np.array(var3),3)
                        if(var5[-1] < var5[-2] and dw['close'].values[-1] > dw['open'].values[-1] and dw['macd'].values[-1] > dw['macd'].values[-2] and (sum(buyVolumes)/len(buyVolumes)) / (sum(sellVolumes)/len(sellVolumes)) > 1.01):
                            result = '买入'
                            ones = '哥要上涨公式'

                    #抄底公式
                    if( dw['MACD_macd'].values[-2] < dw['MACD_macdsignal'].values[-2] and dw['MACD_macd'].values[-1] > dw['MACD_macdsignal'].values[-1] and  dw['MACD_macdsignal'].values[-1] <= -0.07 and dw['macd'].values[-1] > dw['macd'].values[-2]+5):
                        result = '买入'
                        ones = '底部公式'
                break
            except:
                time.sleep(5)
                continue
        print(str(datetime.now())+'--->>>'+ones+'--->>>'+result)
        Datainfo.saveinfo('--->>>'+ones+'--->>>'+result+'--->>>')
        return result


    def getfulldata(df,symbol):


        #获取参数历史数据
        
        # MA - Moving average 移动平均线
        # 函数名：MA
        # 名称： 移动平均线
        # 简介：移动平均线，Moving Average，简称MA，原本的意思是移动平均，由于我们将其制作成线形，所以一般称之为移动平均线，简称均线。它是将某一段时间的收盘价之和除以该周期。 比如日线MA5指5天内的收盘价除以5 。
        # real = MA(close, timeperiod=30, matype=0)
        # 调用talib计算5\35\135日指数移动平均线的值
        

        df['close5'] = ta.EMA(np.array(df['close'].values), timeperiod=5)
        df['close35'] = ta.EMA(np.array(df['close'].values), timeperiod=35)
        df['close135'] = ta.EMA(np.array(df['close'].values), timeperiod=135)

        df["MACD_macd"],df["MACD_macdsignal"],df["MACD_macdhist"] = ta.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=60)
        df['macd'] = 2*(df["MACD_macd"]-df["MACD_macdsignal"])

        df["MA"] = ta.MA(df['close'].values, timeperiod=30, matype=0)
        # EMA和MACD
        df['obv'] = ta.OBV(df['close'].values,df['vol'].values)
        df['maobv'] = ta.MA(df['obv'].values, timeperiod=30, matype=0)

        df['TRIX'] = ta.TRIX(np.array(df['close'].values), timeperiod=14)
        df['MATRIX'] = ta.MA(df['TRIX'].values, timeperiod=30, matype=0)

        
        df.to_csv(f'./datas/okex/'+symbol+'/close.csv',index = False)

       

    #获取用户API信息
    def get_userinfo():

        with open('api.json', 'r', encoding='utf-8') as f:
            obj = json.loads(f.read())


        api_key = obj['api_key']
        secret_key = obj['secret_key']
        passphrase = obj['passphrase']

        # flag是实盘与模拟盘的切换参数 flag is the key parameter which can help you to change between demo and real trading.
        # flag = '1'  # 模拟盘 demo trading
        flag = '0'  # 实盘 real trading

        return api_key,secret_key,passphrase,flag

    #保存信息
    def saveinfo(info):

        f_info = f'./datas/log/infodata.txt'
 

        with open(f_info,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾 
            file.write('\n'+str(info)+str(datetime.now()))

    
    #保存最终信息
    def save_finalinfo(info):

        f_day = f'./datas/log/day_buy.txt'

        with open(f_day,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾 
            file.write('\n'+str(info)+'--->>>'+str(datetime.now()))


    
    #设置自动下单
    def orderbuy(api_key, secret_key, passphrase, flag,symbol):

        # account api
        accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
        
        # 设置持仓模式  Set Position mode
        result = accountAPI.get_position_mode('long_short_mode')
        # 设置杠杆倍数  Set Leverage
        result = accountAPI.set_leverage(instId=symbol.upper()+'-USD-SWAP', lever='100', mgnMode='cross')
        #Datainfo.saveinfo('设置100倍保证金杠杆完毕。。。')
        # trade api
        tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
        # 批量下单  Place Multiple Orders
        # 批量下单  Place Multiple Orders
        result = tradeAPI.place_multiple_orders([
             {'instId': symbol.upper()+'-USD-SWAP', 'tdMode': 'cross', 'side': 'buy', 'ordType': 'market', 'sz': '3',
              'posSide': 'long',
              'clOrdId': 'a12344', 'tag': 'test1210'},
    

         ])
        print(result)

        #Datainfo.saveinfo('下单完毕。。。')

        lastprice = Datainfo.getlastprice(api_key, secret_key, passphrase, flag,symbol)

        #Datainfo.saveinfo('获取最新价格。。。'+str(lastprice))
        
        # 调整保证金  Increase/Decrease margint
        result = accountAPI.Adjustment_margin(symbol.upper()+'-USD-SWAP', 'short', 'add', '5')
        #Datainfo.saveinfo('调整保证金完毕。。。')

        # 策略委托下单  Place Algo Order
        result = tradeAPI.place_algo_order(symbol.upper()+'-USD-SWAP', 'cross', 'sell', ordType='conditional',
                                            sz='3',posSide='long', tpTriggerPx=str(float(lastprice)+100), tpOrdPx=str(float(lastprice)+100))
        #Datainfo.saveinfo(str(datetime.now())+'设置止盈完毕。。。'+str(float(lastprice)+50))


        sendtext = '买入'+symbol.upper()+'-USD-SWAP -->> 3笔，价格是'+str(lastprice)+'，设置止盈完毕。。。'+str(float(lastprice)+100)
        Datainfo.save_finalinfo('买入价格是--》》'+str(lastprice)+'，设置止盈完毕。。。'+str(float(lastprice)+100))
        SendDingding.sender(sendtext)



    #查询最新价格
    def getlastprice(api_key, secret_key, passphrase, flag,symbol):

        # market api
        marketAPI = Market.MarketAPI(api_key, secret_key, passphrase, False, flag)

        # 获取单个产品行情信息  Get Ticker
        result = marketAPI.get_ticker(symbol.upper()+'-USDT-SWAP')
        print(eval(json.dumps(result['data'][0])))
        
        return eval(json.dumps(result['data'][0]))['last']
   

  
    class mywindows_multiprocessing ():

    
        #运行函数 必须写
        def run():
            sch = Datainfo.windowsshow()

            #声明6进程保存数据
            p1 = multiprocessing.Process(target = sch.showwindows)
            p2 = multiprocessing.Process(target = sch.okex5M_buy)
            p3 = multiprocessing.Process(target = sch.okex15M_buy)
            p5 = multiprocessing.Process(target = sch.okex60M_buy)
            p6 = multiprocessing.Process(target = sch.okex240M_buy)

            #6个进程开始运行
            p6.start()
            p5.start()
            p3.start()
            p2.start()
            p1.start()

            p6.join()
            p5.join()
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
            #设置日期最大值与最小值，在当前日期的基础上，后一年与前一年
            self.dateTimeEdit.setMinimumDate(QDate.currentDate().addDays(-365))
            self.dateTimeEdit.setMaximumDate(QDate.currentDate().addDays(365))
            self.dateTimeEdit.setDisplayFormat("yyyy-MM-dd HH:mm:ss ddd a")
            #设置日历控件允许弹出
            self.dateTimeEdit.setCalendarPopup(True)
            self.verticalLayout.addWidget(self.dateTimeEdit)

            self.textBrowserone = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
            self.textBrowserone.setObjectName("textBrowser")
            self.textBrowserone.setStyleSheet("background-color:#4e72b8;font-size:15px; font-family:'行楷';font-weight:bold;")
            self.textBrowserone.setAlignment(Qt.AlignCenter)
            self.textBrowserone.verticalScrollBar().setValue(self.textBrowserone.maximumHeight())
            self.verticalLayout.addWidget(self.textBrowserone)

            self.textBrowsertwo = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
            self.textBrowsertwo.setObjectName("textBrowser")
            self.textBrowsertwo.setStyleSheet("background-color:#9b95c9;font-size:15px; font-family:'行楷';font-weight:bold;")
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
          

            f_day = open(f'./datas/log/day_buy.txt',"r",encoding='utf-8')   #设置文件对象
            day_buy = f_day.read()[-1600:]     #将txt文件的所有内容读入到字符串str中
            f_day.close()   #将文件关闭
            if(day_buy):
                self.textBrowsertwo.clear()
                self.textBrowsertwo.append(day_buy)

            f_info = open(f'./datas/log/infodata.txt',"r",encoding='utf-8')   #设置文件对象
            infodata = f_info.read()[-1600:]     #将txt文件的所有内容读入到字符串str中
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
            MainWindow.setWindowTitle("人工智能程序———okex自动交易v3.0")
            # 加载操作
            dark_stylesheet=qdarkstyle.load_stylesheet_from_environment()# PyQtGraphdark_stylesheet=qdarkstyle.load_stylesheet_from_environment(is_pyqtgraph)
            app.setStyleSheet(dark_stylesheet)
            MainWindow.show() # 执行QMainWindow的show()方法，显示这个QMainWindow
            Datainfo.textBrowserone = ui.textBrowserone
            Datainfo.textBrowsertwo = ui.textBrowsertwo

            ui.textBrowserone.append("欢迎运行 人工智能程序——okex自动交易v3.0")
  
            sys.exit(app.exec_()) # 使用exit()或者点击关闭按钮退出QApp


        def okex5M_buy(self):



            scheduler = BlockingScheduler()
            scheduler.add_job((self.getdatainfo), 'cron', args = ['5'], minute='*/5')
            print(scheduler.get_jobs())
            try:
                scheduler.start()
            except KeyboardInterrupt:
                scheduler.shutdown()

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

            time.sleep(45)
 

            symbollist = ['btc']

            for symbol in symbollist:

                isbuy  =  Datainfo.eth_isbuy(minute,symbol)

                

                if('买入' == isbuy):
                    

                    api_key, secret_key, passphrase, flag = Datainfo.get_userinfo()
                    Datainfo.orderbuy(api_key, secret_key, passphrase, flag,symbol)
   

           
#发钉钉的类先声明
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
                "content": str(datetime.now())+'--->>>我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫 ！^_^     -->> '+sendtexts
            }
        }
        roboturl='https://oapi.dingtalk.com/robot/send?access_token=f8195c9e4ad6da4427d67e80dffed5d07ecaca1d1e79462fb5c0a9c6b12e90f2'
        r.post(roboturl, data=json.dumps(text_data),params=params, headers=headers )

if __name__ == '__main__':

  
    
    #启动
    
  
    print('=============================================================================================')
    print('我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫 ！^_^')
    print('=============================================================================================')
    print('=============================================================================================')


    


    paths = []
    paths.append(f'./datas/okex/')
    paths.append(f'./datas/log/')


    symbollist = ['btc']
    #将txt文件的所有内容读入到字符串str中

    for symbol in symbollist:
    
        paths.append(f'./datas/okex/'+symbol)
    

    for p in paths :
        try:
            os.makedirs(p,exist_ok=True)
        except:
            pass
    
    time.sleep(3)

    
    f_info = f'./datas/log/infodata.txt'
    f_day = f'./datas/log/day_buy.txt'

    #清空文件内容
    file = open(f'./datas/log/day_buy.txt', 'w',encoding='utf-8').close()
    file = open(f'./datas/log/infodata.txt', 'w',encoding='utf-8').close()


    with open(f_info,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾 
        file.write("开始运行okex API 获取数据==="+str(datetime.now()))

    time.sleep(3)

    mywindowsmultiprocessing = Datainfo.mywindows_multiprocessing
    mywindowsmultiprocessing.run()
