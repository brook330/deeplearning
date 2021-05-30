# -*- coding: utf-8 -*-
import threading
import os,time,sys
from apscheduler.schedulers.blocking import BlockingScheduler
import datetime,pyautogui,subprocess
import pandas as pd
from datetime import datetime,timedelta
import numpy as np
import talib as ta
import csv
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import hmac
import hashlib
import base64
import urllib.parse
import requests as  r
import json
import numpy
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



class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return json.JSONEncoder.default(self,obj)


class Datainfo:

    def isbuy():

        time.sleep(15)

        Datainfo.saveinfo('开始获取是否可以买入。。。')

        t = time.time()

        #print (t)                       #原始时间数据
        #print (int(t))                  #秒级时间戳
        #print (int(round(t * 1000)))    #毫秒级时间戳
        #print (int(round(t * 1000000))) #微秒级时间戳
        tt = str((int(t * 1000)))
        ttt = str(int(round(t * 1000)))
    
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
        'referer': 'https://www.okex.com/markets/swap-data/eth-usd',
        'cookie': 'locale=zh_CN; _gcl_au=1.1.1849415495.'+str(tt)+'; _ga=GA1.2.1506507962.'+tt+'; _gid=GA1.2.256681666.'+tt+'; amp_56bf9d=gqC_GMDGl4q5Tk-BJhT-oP...1f6snl1p6.1f6snloll.0.0.0',
        }

        params = (
            ('t', str(tt)),
            ('unitType', '0'),
        )

        response = r.get('https://www.okex.com/v3/futures/pc/market/takerTradeVolume/ETH', headers=headers, params=params)


        response2 = r.get('https://www.okex.com/v3/futures/pc/market/openInterestAndVolume/ETH', headers=headers, params=params)

        if response.cookies.get_dict(): #保持cookie有效 
                s=r.session()
                c = r.cookies.RequestsCookieJar()#定义一个cookie对象
                c.set('cookie-name', 'cookie-value')#增加cookie的值
                s.cookies.update(c)#更新s的cookie
                s.get(url = 'https://www.okex.com/v3/futures/pc/market/takerTradeVolume/ETH?t='+tt+'&unitType=0')

        if response2.cookies.get_dict(): #保持cookie有效 
                s=r.session()
                c = r.cookies.RequestsCookieJar()#定义一个cookie对象
                c.set('cookie-name', 'cookie-value')#增加cookie的值
                s.cookies.update(c)#更新s的cookie
                s.get(url = 'https://www.okex.com/v3/futures/pc/market/openInterestAndVolume/ETH?t='+tt+'&unitType=0')

        df = pd.DataFrame(eval(json.dumps(response.json()))['data'])
        datelist = []
        for timestamp in df['timestamps']:
            datelist.append(datetime.fromtimestamp(timestamp/1000).strftime("%Y-%m-%d %H:%M:%S"))

        df['timestamps'] = pd.to_datetime(datelist)
        df['plot']=numpy.divide(df['buyVolumes'].values.astype(numpy.float64),df['sellVolumes'].values.astype(numpy.float64))
        df = df[['timestamps','buyVolumes','sellVolumes','plot']]

        df3 = pd.DataFrame(eval(json.dumps(response2.json()))['data'])
        df['openInterests'] = df3['openInterests']

        df.to_csv(f'./datas/okex/eth/ethusd.csv',index = False)
        df = pd.read_csv(f'./datas/okex/eth/ethusd.csv')
        result = df['plot'].values[-1]>1 and df['plot'].values[-1] > df['plot'].values[-2] and df['openInterests'].values[-1] > df['openInterests'].values[-2]

        Datainfo.saveinfo('获取数据完毕。。。   判断为： -->>'+str(result)+'   -->>我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫 ！^_^')

        my_file = Path(f'./datas/okex/eth/ethusd_final.csv')

        df2 = df.copy()

        if my_file.exists():
            df2 = pd.read_csv(f'./datas/okex/eth/ethusd_final.csv')
            #增加一行 append
            df2.append(df.iloc[-1:])        
        
        df2.to_csv(f'./datas/okex/eth/ethusd_final.csv',index = False)
        Datainfo.saveinfo('保存所有的 ethusd 数据完毕。。。   ')
        return  result

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

        Datainfo.saveinfo('获取API_KEY。。。')

        

        return api_key,secret_key,passphrase,flag

    #保存信息
    def saveinfo(info):

        f_info = f'./datas/log/infodata.txt'
 

        with open(f_info,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾 
            file.write('\n'+info+str(datetime.now()))

    
    #保存最终信息
    def save_finalinfo(info):

        f_day = f'./datas/log/day_buy.txt'

        with open(f_day,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾 
            file.write('\n'+info+str(datetime.now()))


    
    #设置自动下单
    def orderbuy(api_key, secret_key, passphrase, flag):

        # account api
        accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
        
        # 设置持仓模式  Set Position mode
        result = accountAPI.get_position_mode('long_short_mode')
        # 设置杠杆倍数  Set Leverage
        result = accountAPI.set_leverage(instId='ETH-USD-SWAP', lever='100', mgnMode='cross')
        Datainfo.saveinfo('设置100倍保证金杠杆完毕。。。')
        # trade api
        tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
        # 批量下单  Place Multiple Orders
        result = tradeAPI.place_multiple_orders([
             {'instId': 'ETH-USD-SWAP', 'tdMode': 'cross', 'side': 'buy', 'ordType': 'market', 'sz': '5',
              'posSide': 'long',
              'clOrdId': 'a12344', 'tag': 'test1210'},

         ])
        print(result)

        Datainfo.saveinfo('下单完毕。。。')

        lastprice = Datainfo.getlastprice(api_key, secret_key, passphrase, flag)

        Datainfo.saveinfo('获取最新价格。。。'+str(lastprice))
        
        # 调整保证金  Increase/Decrease margint
        result = accountAPI.Adjustment_margin('ETH-USD-SWAP', 'short', 'add', '10')
        Datainfo.saveinfo('调整保证金完毕。。。')

        # 策略委托下单  Place Algo Order
        result = tradeAPI.place_algo_order('ETH-USD-SWAP', 'cross', 'sell', ordType='conditional',
                                            sz='5',posSide='long', tpTriggerPx=str(float(lastprice)+10), tpOrdPx=str(float(lastprice)+9))
        Datainfo.saveinfo('设置止盈完毕。。。'+str(float(lastprice)+10))

        sendtext = '100倍杠杆，全仓委托：ETH-USD-SWAP -->> 5笔，价格是'+str(lastprice)+'，设置止盈完毕。。。'+str(float(lastprice)+10)
        Datainfo.save_finalinfo('我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫 ！^_^     -->>'+sendtext)
        SendDingding.sender(sendtext)

    #查询最新价格
    def getlastprice(api_key, secret_key, passphrase, flag):

        # market api
        marketAPI = Market.MarketAPI(api_key, secret_key, passphrase, False, flag)

        # 获取单个产品行情信息  Get Ticker
        result = marketAPI.get_ticker('ETH-USDT-SWAP')
        
        return eval(json.dumps(result['data'][0]))['last']
   
  
    class mywindows_multiprocessing ():

    
        #运行函数 必须写
        def run():
            sch = Datainfo.windowsshow()

            #声明2线程保存数据
            p1 = multiprocessing.Process(target = sch.showwindows)
            p2 = multiprocessing.Process(target = sch.okex5M_buy)
            

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
          

            f_day = open(f'./datas/log/day_buy.txt',"r",encoding='utf-8')   #设置文件对象
            day_buy = f_day.read()     #将txt文件的所有内容读入到字符串str中
            f_day.close()   #将文件关闭
            if(day_buy):
                self.textBrowsertwo.clear()
                self.textBrowsertwo.append(day_buy)

            f_info = open(f'./datas/log/infodata.txt',"r",encoding='utf-8')   #设置文件对象
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
            scheduler.add_job((self.getdatainfo), 'cron', minute='*/5')
            print(scheduler.get_jobs())
            try:
                scheduler.start()
            except KeyboardInterrupt:
                scheduler.shutdown()
        
        def getdatainfo(self):

            api_key, secret_key, passphrase, flag = Datainfo.get_userinfo()
            # market api
            marketAPI = Market.MarketAPI(api_key, secret_key, passphrase, False, flag)
            result = marketAPI.get_candlesticks('ETH-USD-SWAP', bar='5m')

            df = pd.DataFrame(result['data'])
            df.columns = ['date','open','high','low','close','vol','p']
            df = df.iloc[::-1]
                
            if(df['close'].values[-1] < df['open'].values[-1]):
                Datainfo.saveinfo('下跌趋势，不买人，直接返回。。。')
                return 0

            Datainfo.saveinfo('上升趋势，继续观察。。。')
            if(Datainfo.isbuy()):

                Datainfo.orderbuy(api_key, secret_key, passphrase, flag)



           
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
                "content": '我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫 ！^_^     -->> '+sendtexts
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
    paths.append(f'./datas/okex/eth')        
    paths.append(f'./datas/log/')

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

    

        
