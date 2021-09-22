# -*- coding: gbk -*-

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

                time.sleep(minute)

                t = time.time()

                #print (t)                       #ԭʼʱ������
                #print (int(t))                  #�뼶ʱ���
                #print (int(round(t * 1000)))    #���뼶ʱ���
                #print (int(round(t * 1000000))) #΢�뼶ʱ���
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

                time.sleep(minute)

                response = r.get('https://gu.sina.cn/ft/api/jsonp.php/var%5E%%5E20_'+symbol+'_'+str(minute)+'_'+ttt+'=/GlobalService.getMink', headers=headers, params=params ,timeout=1)

                if response.cookies.get_dict(): #����cookie��Ч 
                    s=r.session()
                    c = r.cookies.RequestsCookieJar()#����һ��cookie����
                    c.set('cookie-name', 'cookie-value')#����cookie��ֵ
                    s.cookies.update(c)#����s��cookie
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
                df = self.getfulldata(df)

                time.sleep(minute)

                return df
                break
            except:
                time.sleep(5)
                continue
    

    def getfulldata(self,df):
        #��ȡ������ʷ����
       
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

        return df

    def buyinfo(self,df,symbol,minute):
        
        #print("\nʱ���ǣ�"+str(df['date'][-1:].values)+"��ʼ��ȡ�Ƿ�"+symbol+"�����ź� ����")
        f_info = f'./datas/infodata.txt'
        with open(f_info,"a+",encoding='utf-8') as file:   #a :   д���ļ������ļ�����������ȴ�����д�룬�����Ḳ��ԭ�ļ�������׷�����ļ�ĩβ
            file.write("\nʱ���ǣ�"+str(df['date'][-1:].values)+"��ʼ��ȡ�Ƿ�"+symbol+"�����ź� ����")
        

        dw=pd.DataFrame()
        dw=df
        symbol = SendDingding.get_symbol_name(symbol)
        X1 = dw['close'].values[-1]/dw['volume'].values[-1]/dw['MA'].values[-1]*dw['obv'].values[-1]/dw['maobv'].values[-1]*dw['TRIX'].values[-1]*dw['MATRIX'].values[-1]*dw['close5'].values[-1]/dw['close135'].values[-1]*dw['macd'].values[-1]
        X2 = dw['close'].values[-2]/dw['volume'].values[-2]/dw['MA'].values[-2]*dw['obv'].values[-2]/dw['maobv'].values[-2]*dw['TRIX'].values[-2]*dw['MATRIX'].values[-2]*dw['close5'].values[-2]/dw['close135'].values[-2]*dw['macd'].values[-2]

        Y1 = dw['close'].values[-1]*float(dw['MATRIX'].values[-1])*float(dw['TRIX'].values[-1])
        Y2 = dw['close'].values[-2]*float(dw['MATRIX'].values[-2])*float(dw['TRIX'].values[-2])

        maxvalue = dw.iloc[-50:][(dw['macd'] == dw['macd'][-50:].max())]['close']
        minvalue = dw.iloc[-50:][(dw['macd'] == dw['macd'][-50:].min())]['close']
        value = maxvalue.values - minvalue.values
        value_618 = maxvalue.values - value * 0.618
        value_192 = maxvalue.values - value * 0.192

        VAR1 = ta.EMA(np.array(ta.EMA(np.array(df['macd'].values), timeperiod=9)), timeperiod=9)
        kongpan = (VAR1[1:]-VAR1[:-1])/VAR1[:-1]*1000
        ref_kongpan = (VAR1[2:]-VAR1[:-2])/VAR1[:-2]*1000
        
        
        if(not(X1 >5 and X2 < -3) and X1 >0 and X2 <0 and not(Y1 >0 and Y2 < 0) and dw['macd'].values[-1] > dw['macd'].values[-2] +5):
            print("\n����"+str(minute)+'���ӣ�'+symbol+"����������������ʱ���ǣ�"+str(df['date'][-1:].values[0])+"������ֵ�ǣ�"+str(df['close'][-1:].values),symbol)
            SendDingding.buy_sender("\n����"+str(minute)+'���ӣ�'+symbol+"����������������ʱ���ǣ�"+str(df['date'][-1:].values[0])+"������ֵ�ǣ�"+str(df['close'][-1:].values),symbol)
        elif(dw['macd'].values[-1] > dw['macd'].values[-2] +5 and  kongpan[-1:]>ref_kongpan[-1:] and kongpan[-1:]<50 and kongpan[-1:]>0 and (df['macd'].values[-1]>df['macd'].values[-2] or df['macd'].values[-2]>df['macd'].values[-3])):
            print("\n����"+str(minute)+'���ӣ�'+symbol+"����������������ʱ���ǣ�"+str(df['date'][-1:].values[0])+"������ֵ�ǣ�"+str(df['close'][-1:].values),symbol)
            SendDingding.buy_sender("\n����"+str(minute)+'���ӣ�'+symbol+"����������������ʱ���ǣ�"+str(df['date'][-1:].values[0])+"������ֵ�ǣ�"+str(df['close'][-1:].values),symbol)
        elif(dw['macd'].values[-1] > dw['macd'].values[-2] +5 and  dw['close'].values[-1] > value_618 and dw['close'].values[-1] < value_192  and dw['macd'].values[-1] > dw['macd'].values[-2] and dw['volume'].values[-1] > dw['volume'].values[-2]*1.01):
            print("\n����"+str(minute)+'���ӣ�'+symbol+"����������������ʱ���ǣ�"+str(df['date'][-1:].values[0])+"������ֵ�ǣ�"+str(df['close'][-1:].values),symbol)
            SendDingding.buy_sender("\n����"+str(minute)+'���ӣ�'+symbol+"����������������ʱ���ǣ�"+str(df['date'][-1:].values[0])+"������ֵ�ǣ�"+str(df['close'][-1:].values),symbol)
          
        elif(dw['macd'].values[-1] > dw['macd'].values[-2] +5 and  dw['macd'].values[-2] == dw['macd'][-40:].min() and dw['macd'].values[-2] < 0 and dw['macd'].values[-2] < dw['macd'].values[-1]):
            print("\n����"+str(minute)+'���ӣ�'+symbol+"����������������ʱ���ǣ�"+str(df['date'][-1:].values[0])+"������ֵ�ǣ�"+str(df['close'][-1:].values),symbol)
            SendDingding.buy_sender("\n����"+str(minute)+'���ӣ�'+symbol+"����������������ʱ���ǣ�"+str(df['date'][-1:].values[0])+"������ֵ�ǣ�"+str(df['close'][-1:].values),symbol)
        else:
            print(str(minute)+'����'+str(symbol)+'����������')
       

    #�ж�ʱ������
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

    #NQ,CL,YM,HSI,ES,NK,CHA50CFD 5����


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

    
    #NQ,CL,YM,HSI,ES,NK,CHA50CFD  15����
    
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

  


#����������������
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
        
        
        sendtexts = "����ʱ�䣺 " +today+"��\n �ﵽĿ������ֵ��׼�����롣������ڻ��ǣ�"+symbol+"��\n���ڵ����̼��ǣ�"+str(close)+"\n"+ "�������룬\n�������ػ��ߣ�Ҳ��һȺʱ�̶Կ�Σ�պͷ��Ŀ����棡����"

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
        
        
        sendtexts = "����ʱ�䣺 " +today+"��\n �ﵽĿ������ֵ��׼���������������ڻ��ǣ�"+symbol+"��\n���ڵ����̼��ǣ�"+str(close)+"\n"+ "����������\n�������ػ��ߣ�Ҳ��һȺʱ�̶Կ�Σ�պͷ��Ŀ����棡����"

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
            symbol = '��˹���ָ���ڻ�'
        elif(symbol == 'CL'):
            symbol = '����ԭ��'
        elif(symbol == 'YM'):
            symbol = '����˹ָ���ڻ�'
        elif(symbol == 'HSI'):
            symbol = '����ָ���ڻ�'
        elif(symbol == 'ES'):
            symbol = '����500ָ���ڻ�'
        elif(symbol == 'NK'):
            symbol = '�վ�225ָ���ڻ�'
        elif(symbol == 'CHA50CFD'):
            symbol = '��ʱ�й�A50ָ���ڻ�'



        return symbol

if __name__ == '__main__':


    f_info = f'./datas/infodata.txt'

    try:
        os.makedirs(f'./datas',exist_ok=True)
    except:
        pass

    time.sleep(0.5)

    #����ļ�����
    file = open(f'./datas/infodata.txt', 'w',encoding='utf-8').close()
    time.sleep(0.5)


    dingzhi =  RuntimeScheduler()

    #�������

    p1 =multiprocessing.Process(target = dingzhi.get_final_NQ_5M_job)
    p2 =multiprocessing.Process(target = dingzhi.get_final_NQ_15M_job)
    p3 =multiprocessing.Process(target = dingzhi.get_final_YM_5M_job)
    p4 =multiprocessing.Process(target = dingzhi.get_final_YM_15M_job)
    p5 =multiprocessing.Process(target = dingzhi.get_final_HSI_5M_job)
    p6 =multiprocessing.Process(target = dingzhi.get_final_HSI_15M_job)
    p7 =multiprocessing.Process(target = dingzhi.get_final_ES_5M_job)
    p8 =multiprocessing.Process(target = dingzhi.get_final_ES_15M_job)
    p9 =multiprocessing.Process(target = dingzhi.get_final_NK_5M_job)
    p10 =multiprocessing.Process(target = dingzhi.get_final_NK_15M_job)
    p11 =multiprocessing.Process(target = dingzhi.get_final_CHA50CFD_5M_job)
    p12 =multiprocessing.Process(target = dingzhi.get_final_CHA50CFD_15M_job)

    #start

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


    #join

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
