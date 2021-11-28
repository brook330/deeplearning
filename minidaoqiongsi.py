# -*- coding: utf-8 -*-
import requests as r
import pandas as pd
from datetime import datetime,timedelta
import time,os,shutil
import talib as ta
import hmac
import hashlib
import base64
import urllib.parse
import json,math
from apscheduler.schedulers.blocking import BlockingScheduler
import multiprocessing
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('mode.chained_assignment', None)

class NQ_Datainfo():

   
    def getdatainfo_minute(self,minute,symbol):

        for i in range(10000):

            try:

                time.sleep(minute/100)

                t = time.time()

                #print (t)                       #原始时间数据
                #print (int(t))                  #秒级时间戳
                #print (int(round(t * 1000)))    #毫秒级时间戳
                #print (int(round(t * 1000000))) #微秒级时间戳
                tt = str((int(t * 1000)))
                ttt = str(int(round(t * 1000)))

                headers = {
                    'authority': 'gu.sina.cn',
                    'sec-ch-ua': '^\\^Google',
                    'sec-ch-ua-mobile': '?0',
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
                    'accept': '*/*',
                    'sec-fetch-site': 'cross-site',
                    'sec-fetch-mode': 'no-cors',
                    'sec-fetch-dest': 'script',
                    'referer': 'https://finance.sina.com.cn/futures/quotes/'+symbol+'.shtml',
                    'accept-language': 'zh-CN,zh;q=0.9',
                    'cookie': 'ustat=__115.206.97.215_'+tt+'_0.65452100; genTime='+tt+'; vt=4; QUOTES-SINA-CN=',
                }

                params = (
                    ('symbol', symbol),
                    ('type', str(minute)),
                )

                time.sleep(minute/100)

                response = r.get('https://gu.sina.cn/ft/api/jsonp.php/var%5E%%5E20_'+symbol+'_'+str(minute)+'_'+ttt+'=/GlobalService.getMink', headers=headers, params=params ,timeout=1)

                if response.cookies.get_dict(): #保持cookie有效 
                    s=r.session()
                    c = r.cookies.RequestsCookieJar()#定义一个cookie对象
                    c.set('cookie-name', 'cookie-value')#增加cookie的值
                    s.cookies.update(c)#更新s的cookie
                    s.get(url='https://gu.sina.cn/ft/api/jsonp.php/var%5E%%5E20_'+symbol+'_'+str(minute)+'_'+ttt+'=/GlobalService.getMink')

                s =response.text
                s = s.strip('/*<script>location.href=\'//sina.com\';</script>*/\nvar^%^20_'+symbol+'_'+str(minute)+'_'+ttt+'=([')
                s = s.strip('])')
                values = eval(s)
                df = pd.DataFrame(list(values))
                df.columns=['date','open','high','low','close','volume','p']
                df['volume'] = df['volume'].astype('float')
                df.to_csv(symbol+str(minute)+'.csv',index=False)
                df = pd.read_csv(symbol+str(minute)+'.csv')
                df['volume'].values[-1]
                res = pd.read_csv(symbol+'-'+minute+'min.csv')
                res = res.append([{'date':df['date'].values[-1],'open':df['open'].values[-1],'high':df['high'].values[-1],'low':df['low'].values[-1],'close':df['close'].values[-1],'volume':df['volume'].values[-1]}], ignore_index=True)
                df['volume']=df['volume'].astype(np.float64)
                df['close']=df['close'].astype(np.float64)
                df['open']=df['open'].astype(np.float64)
                df['high']=df['high'].astype(np.float64)
                df['low']=df['low'].astype(np.float64)
                res.to_csv(symbol+'-'+minute+'min.csv')
                df = self.getfulldata(df)

                time.sleep(minute/100)

                return df
                break
            except:
                time.sleep(5/100)
                continue
    

    def getfulldata(self,df):
        #获取参数历史数据
       
        df['close5'] = ta.EMA(np.array(df['close'].values), timeperiod=5)
        df['close35'] = ta.EMA(np.array(df['close'].values), timeperiod=35)
        df['close135'] = ta.EMA(np.array(df['close'].values), timeperiod=135)

        df["MACD_macd"],df["MACD_macdsignal"],df["MACD_macdhist"] = ta.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=3)
        df['macd'] = 2*(df["MACD_macd"]-df["MACD_macdsignal"])

        df["MA"] = ta.MA(df['close'].values, timeperiod=30, matype=0)
        # EMA和MACD
        df['obv'] = ta.OBV(df['close'].values,df['volume'].values)
        df['maobv'] = ta.MA(df['obv'].values, timeperiod=30, matype=0)

        df['TRIX'] = ta.TRIX(np.array(df['close'].values), timeperiod=14)
        df['MATRIX'] = ta.MA(df['TRIX'].values, timeperiod=30, matype=0)

        return df

    def buyinfo(self,df,symbol,minute):
        
        #print("\n时间是："+str(df['date'][-1:].values)+"开始获取是否"+symbol+"买入信号 运算")
        f_info = f'./datas/infodata.txt'
        with open(f_info,"a+",encoding='utf-8') as file:   #a :   写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾
            file.write("\n时间是："+str(df['date'][-1:].values)+"开始获取是否"+symbol+"买入信号 运算")
        

        dw=pd.DataFrame()
        dw=df
        symbol = SendDingding.get_symbol_name(symbol)

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

        

        #===判断是否买入或者卖出
        if(KONGPAN1>0 and KONGPAN2>0 and X1>0 and X2>0 and Y1>0 and Y2>0 and dw['macd'].values[-1]>0):           
                
                        
            if(KONGPAN1>KONGPAN2 and  X1>X2 and Y1>Y2 and dw['macd'].values[-1] > dw['macd'].values[-2]+3):  
                print("\n买入"+str(minute)+'分钟，'+symbol+"，符合条件，买入时间是："+str(df['date'][-1:].values[0])+"，买入值是："+str(df['close'][-1:].values),symbol)
                SendDingding.buy_sender("\n买入"+str(minute)+'分钟，'+symbol+"，符合条件，买入时间是："+str(df['date'][-1:].values[0])+"，买入值是："+str(df['close'][-1:].values),symbol)
            else:
                print("\n"+str(df['date'][-1:].values[0])+"--->>>"+str(minute)+'分钟，--->>>'+symbol+"--->>>不满足条件")

       

    #判断时间买卖
    def timepass(self):
        if(datetime.now().weekday()+1!=7):

                if((datetime.now().weekday()+1 >1 and datetime.now().weekday()+1<=6 and datetime.now() >= datetime.strptime(str(datetime.now().date())+' 00:00:00', '%Y-%m-%d %H:%M:%S') and datetime.now() <= datetime.strptime(str(datetime.now().date())+' 05:00:00', '%Y-%m-%d %H:%M:%S'))  or  (datetime.now().weekday()+1<=5 and datetime.now() >= datetime.strptime(str(datetime.now().date())+' 06:00:00', '%Y-%m-%d %H:%M:%S') and datetime.now() <= datetime.strptime(str(datetime.now().date())+' 23:59:59', '%Y-%m-%d %H:%M:%S'))):
                    return True
                else:
                    return False
        else:
            return False

    def time_hsi_quantian(self):

        if((datetime.now().weekday()+1 >1 and datetime.now().weekday()+1<=6 and datetime.now() >= datetime.strptime(str(datetime.now().date())+' 00:00:00', '%Y-%m-%d %H:%M:%S') and datetime.now() <= datetime.strptime(str(datetime.now().date())+' 03:00:00', '%Y-%m-%d %H:%M:%S'))  or  (datetime.now().weekday()+1<=5 and datetime.now() >= datetime.strptime(str(datetime.now().date())+' 09:15:00', '%Y-%m-%d %H:%M:%S') and datetime.now() <= datetime.strptime(str(datetime.now().date())+' 12:00:00', '%Y-%m-%d %H:%M:%S'))  or (datetime.now().weekday()+1<=5 and datetime.now() >= datetime.strptime(str(datetime.now().date())+' 13:00:00', '%Y-%m-%d %H:%M:%S') and datetime.now() <= datetime.strptime(str(datetime.now().date())+' 16:30:00', '%Y-%m-%d %H:%M:%S')) or (datetime.now().weekday()+1<=5 and datetime.now() >= datetime.strptime(str(datetime.now().date())+' 17:15:00', '%Y-%m-%d %H:%M:%S') and datetime.now() <= datetime.strptime(str(datetime.now().date())+' 23:59:59', '%Y-%m-%d %H:%M:%S'))):
            return True
        else:
            return False

    def time_if_part_tian(self):

        if((datetime.now().weekday()+1 >1 and datetime.now().weekday()+1<=6 and datetime.now() >= datetime.strptime(str(datetime.now().date())+' 09:30:00', '%Y-%m-%d %H:%M:%S') and datetime.now() <= datetime.strptime(str(datetime.now().date())+' 11:30:00', '%Y-%m-%d %H:%M:%S'))  or  (datetime.now().weekday()+1<=5 and datetime.now() >= datetime.strptime(str(datetime.now().date())+' 13:00:00', '%Y-%m-%d %H:%M:%S') and datetime.now() <= datetime.strptime(str(datetime.now().date())+' 15:00:00', '%Y-%m-%d %H:%M:%S'))):
            return True
        else:
            return False

    #NQ,CL,YM,HSI,ES,NK,CHA50CFD 5分钟



    def final_NQ_5M(self):
        if(self.timepass()):
            df = self.getdatainfo_minute(5,'NQ')
            self.buyinfo(df,'NQ',5)

    def final_YM_5M(self):
        if(self.timepass()):
            df = self.getdatainfo_minute(5,'YM')
            self.buyinfo(df,'YM',5)

    def final_HSI_5M(self):
        if(self.time_hsi_quantian()):
            df = self.getdatainfo_minute(5,'HSI')
            self.buyinfo(df,'HSI',5)

    def final_ES_5M(self):
        if(self.timepass()):
            df = self.getdatainfo_minute(5,'ES')
            self.buyinfo(df,'ES',5)

    def final_NK_5M(self):
        if(self.timepass()):
            df = self.getdatainfo_minute(5,'NK')
            self.buyinfo(df,'NK',5)

    def final_CHA50CFD_5M(self):
        if(self.time_if_part_tian()):
            df = self.getdatainfo_minute(5,'CHA50CFD')
            self.buyinfo(df,'CHA50CFD',5)

    
    #NQ,CL,YM,HSI,ES,NK,CHA50CFD  15分钟
    
    def final_NQ_15M(self):
        if(self.timepass()):
            df = self.getdatainfo_minute(15,'NQ')
            self.buyinfo(df,'NQ',15)

    def final_YM_15M(self):
        if(self.timepass()):
            df = self.getdatainfo_minute(15,'YM')
            self.buyinfo(df,'YM',15)

    def final_HSI_15M(self):
        if(self.time_hsi_quantian()):
            df = self.getdatainfo_minute(15,'HSI')
            self.buyinfo(df,'HSI',15)

    def final_ES_15M(self):
        if(self.timepass()):
            df = self.getdatainfo_minute(15,'ES')
            self.buyinfo(df,'ES',15)

    def final_NK_15M(self):
        if(self.timepass()):
            df = self.getdatainfo_minute(15,'NK')
            self.buyinfo(df,'NK',15)

    def final_CHA50CFD_15M(self):
        if(self.time_if_part_tian()):
            df = self.getdatainfo_minute(15,'CHA50CFD')
            self.buyinfo(df,'CHA50CFD',15)


class RuntimeScheduler:


    #NQ,CL,YM,HSI,ES,NK,CHA50CFD

    def get_final_NQ_5M_job(self):

        
        nqdata = NQ_Datainfo()
        scheduler = BlockingScheduler()
        scheduler.add_job((nqdata.final_NQ_5M), 'cron', minute='*/5')
        print(scheduler.get_jobs())
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown()
        

    def get_final_YM_5M_job(self):

        
        nqdata = NQ_Datainfo()
        scheduler = BlockingScheduler()
        scheduler.add_job((nqdata.final_YM_5M), 'cron', minute='*/5')
        print(scheduler.get_jobs())
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown()

    def get_final_HSI_5M_job(self):

        
        nqdata = NQ_Datainfo()
        scheduler = BlockingScheduler()
        scheduler.add_job((nqdata.final_HSI_5M), 'cron', minute='*/5')
        print(scheduler.get_jobs())
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown()

    def get_final_ES_5M_job(self):

        
        nqdata = NQ_Datainfo()
        scheduler = BlockingScheduler()
        scheduler.add_job((nqdata.final_ES_5M), 'cron', minute='*/5')
        print(scheduler.get_jobs())
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown()

    def get_final_NK_5M_job(self):

        
        nqdata = NQ_Datainfo()
        scheduler = BlockingScheduler()
        scheduler.add_job((nqdata.final_NK_5M), 'cron', minute='*/5')
        print(scheduler.get_jobs())
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown()

    def get_final_CHA50CFD_5M_job(self):

        
        nqdata = NQ_Datainfo()
        scheduler = BlockingScheduler()
        scheduler.add_job((nqdata.final_CHA50CFD_5M), 'cron', minute='*/5')
        print(scheduler.get_jobs())
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown()



    #=============================NQ,CL,YM,HSI,ES,NK,CHA50CFD==================================================

    def get_final_NQ_15M_job(self):

        
        nqdata = NQ_Datainfo()
        scheduler = BlockingScheduler()
        scheduler.add_job((nqdata.final_NQ_15M), 'cron', minute='*/15')
        print(scheduler.get_jobs())
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown()

    def get_final_YM_15M_job(self):

        
        nqdata = NQ_Datainfo()
        scheduler = BlockingScheduler()
        scheduler.add_job((nqdata.final_YM_15M), 'cron', minute='*/15')
        print(scheduler.get_jobs())
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown()

    def get_final_HSI_15M_job(self):

        
        nqdata = NQ_Datainfo()
        scheduler = BlockingScheduler()
        scheduler.add_job((nqdata.final_HSI_15M), 'cron', minute='*/15')
        print(scheduler.get_jobs())
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown()

    def get_final_ES_15M_job(self):

        
        nqdata = NQ_Datainfo()
        scheduler = BlockingScheduler()
        scheduler.add_job((nqdata.final_ES_15M), 'cron', minute='*/15')
        print(scheduler.get_jobs())
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown()

    def get_final_NK_15M_job(self):

        
        nqdata = NQ_Datainfo()
        scheduler = BlockingScheduler()
        scheduler.add_job((nqdata.final_NK_15M), 'cron', minute='*/15')
        print(scheduler.get_jobs())
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown()

    def get_final_CHA50CFD_15M_job(self):

        
        nqdata = NQ_Datainfo()
        scheduler = BlockingScheduler()
        scheduler.add_job((nqdata.final_CHA50CFD_15M), 'cron', minute='*/15')
        print(scheduler.get_jobs())
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown()

  


#发钉钉的类先声明
class SendDingding:
    def buy_sender(close,symbol):
             
        symbol = SendDingding.get_symbol_name(symbol)

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
        
        
        sendtexts = "本地时间： " +today+"，\n 达到目标买入值，准备买入。买入的期货是："+symbol+"，\n现在的收盘价是："+str(close)+"\n"+ "马上买入，\n我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫！！！"

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
        r.post(roboturl, data=json.dumps(text_data),params=params, headers=headers )

    def sale_sender(close,symbol):
        
        symbol = SendDingding.get_symbol_name(symbol)

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
        
        
        sendtexts = "本地时间： " +today+"，\n 达到目标卖出值，准备卖出。卖出的期货是："+symbol+"，\n现在的收盘价是："+str(close)+"\n"+ "马上卖出，\n我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫！！！"

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
        r.post(roboturl, data=json.dumps(text_data),params=params, headers=headers )

    def get_symbol_name(symbol):
        #NQ,CL,YM,HSI,ES,NK,CHA50CFD
        if(symbol == 'NQ'):
            symbol = '纳斯达克指数期货'
        elif(symbol == 'CL'):
            symbol = '美国原油'
        elif(symbol == 'YM'):
            symbol = '道琼斯指数期货'
        elif(symbol == 'HSI'):
            symbol = '恒生指数期货'
        elif(symbol == 'ES'):
            symbol = '标普500指数期货'
        elif(symbol == 'NK'):
            symbol = '日经225指数期货'
        elif(symbol == 'CHA50CFD'):
            symbol = '富时中国A50指数期货'



        return symbol

if __name__ == '__main__':


    f_info = f'./datas/infodata.txt'

    try:
        os.makedirs(f'./datas',exist_ok=True)
    except:
        pass

    time.sleep(0.5)

    #清空文件内容
    file = open(f'./datas/infodata.txt', 'w',encoding='utf-8').close()
    time.sleep(0.5)

    res = pd.DataFrame(columns=('timestamps','close', 'KONGPAN1', 'KONGPAN2','X1', 'X2', 'Y1' ,'Y2', 'macd1','macd2'))
    
    pathlist = []
    pathlist.append(f'./datas/okex/BTC-USD-SWAP/1minute/1minute.csv')
    pathlist.append(f'./datas/okex/BTC-USD-SWAP/3minute/3minute.csv')
    pathlist.append(f'./datas/okex/BTC-USD-SWAP/5minute/5minute.csv')
    pathlist.append(f'./datas/okex/BTC-USD-SWAP/15minute/15minute.csv')
    pathlist.append(f'./datas/okex/BTC-USD-SWAP/60minute/60minute.csv')
    pathlist.append(f'./datas/okex/BTC-USD-SWAP/240minute/240minute.csv')

    count = 0
    for path in pathlist:
        
        minute = 0

        if(count==0):
            minute = 1
        elif(count==1):
            minute = 3
        elif(count==2):
            minute = 5
        elif(count==3):
            minute = 15
        elif(count==4):
            minute = 60
        elif(count==5):
            minute = 240

        if(not os.path.exists(path)):

            p = f'./datas/okex//'+str(minute)+'minute/'
            try:
                os.makedirs(p,exist_ok=True)
            except:
                pass
            res.to_csv(path,index=0)
        count+=1

    dingzhi =  RuntimeScheduler()

    #定义进程
    p1 =multiprocessing.Process(target = dingzhi.get_final_YM_5M_job)
    p2 =multiprocessing.Process(target = dingzhi.get_final_YM_15M_job)
    p3 =multiprocessing.Process(target = dingzhi.get_final_NQ_5M_job)
    p4 =multiprocessing.Process(target = dingzhi.get_final_NQ_15M_job)
    p1 =multiprocessing.Process(target = dingzhi.get_final_ES_5M_job)
    p2 =multiprocessing.Process(target = dingzhi.get_final_ES_15M_job)
    p3 =multiprocessing.Process(target = dingzhi.get_final_NK_5M_job)
    p4 =multiprocessing.Process(target = dingzhi.get_final_NK_15M_job)

    #start

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()

    #join

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()

