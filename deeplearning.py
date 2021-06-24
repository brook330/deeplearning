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
import talib as ta
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from torch.utils import data
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import multiprocessing
from yacs.config import CfgNode as CN

_C = CN()
_C.DATASETS = CN()

# TRAIN
_C.BATCH_SIZE = 500
#数据N次循环
_C.EPOCHS = 500000000
_C.PRETRAINED_MOEDLS = ''
_C.EXP = CN()
_C.EXP.PATH = 'datas'
_C.EXP.NAME = 'training'
cfg = _C
cfg.freeze()


class CsvDataset(data.Dataset):

    def __init__(self, path, step):
        inputs = []
        outputs = []
        with open(path, 'r', encoding='UTF-8') as (f):
            f_c = csv.reader(f)
            idx = 0
            for row in f_c:
                if idx == 0:
                    idx += 1
                    continue
                tmp = row[1:]
                tmp = [float(i) for i in tmp]
                outputs.append(tmp.pop(0))
                inputs.append(tmp)
                idx += 1

        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)[:, np.newaxis]
        self.step = step
        self.in_max = np.abs(self.inputs).max(0)
        self.out_max = np.abs(self.outputs).max(0)
        #print(self.in_max)
        #print(self.out_max)

    def __len__(self):
        return self.inputs.shape[0] - self.step - 1

    def __getitem__(self, index):
        _input = self.inputs[index:index + self.step]
        output = self.outputs[index + self.step + 1:index + self.step + 2]
        _input = _input / self.in_max
        output = output / self.out_max
        _input = torch.Tensor(_input)
        output = torch.Tensor(output)
        _input = _input.reshape(-1)
        output = output.reshape(-1)
        return (_input, output)


class Test_CsvDataset(data.Dataset):

    
    def __init__(self, path, step):


        inputs = []
        outputs = []
        time = []
        with open(path, 'r', encoding='gbk') as (f):
            f_c = csv.reader(f)
            idx = 0
            for row in f_c:
                if idx == 0:
                    idx += 1
                    continue
                tmp = row[1:]
                tmp = [float(i) for i in tmp]
                tmp = [float(i) for i in tmp]
                time.append(row[:1])
                outputs.append(tmp.pop(0))
                inputs.append(tmp)
                idx += 1


        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)[:, np.newaxis]
        self.time = time
        self.step = step
        self.in_max = np.abs(self.inputs).max(0)
        self.out_max = np.abs(self.outputs).max(0)
        #print(self.outputs.shape)
        self.inputs = self.inputs[-self.step:]
        self.outputs = self.outputs[-1:]
        self.time = self.time[-1:]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        #print(np.transpose(self.inputs[0][0]))
        #print(np.transpose(self.outputs))
        _input = self.inputs[index:index + self.step]
        output = self.outputs
        _input = _input / self.in_max
        output = output / self.out_max
        _input = torch.Tensor(_input)
        output = torch.Tensor(output)
        _input = _input.reshape(-1)
        output = output.reshape(-1)
        #print((_input.cpu()[0] * self.in_max)[:9])
        #print((output.cpu() * self.out_max)[:9])
        return (_input, output, self.out_max, self.time)

class train_net(nn.Module):

    def __init__(self):
        super(train_net, self).__init__()
        self.fc1 = nn.Linear(36*9, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 256)
        self.fc10 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)
        x = F.relu(x)
        x = self.fc10(x)
        x = F.relu(x)
        return x

    def gettrain(self):


                
        Datainfo.saveinfo("开始建立模型")

        device = torch.device("cuda")
        train_dataset = CsvDataset(f'./datas/okex/eth/ethclose.csv', 9)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=500,
          shuffle=True,
          num_workers=0)
        train_steps = len(train_loader)
        net = train_net.cuda(self)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam((train_net.parameters(self)), lr=0.001, betas=(0.9, 0.999), eps=10e-08, weight_decay=0, amsgrad=False)

        Datainfo.saveinfo('=============开始进行每1000轮的预测，最终取loss_item<5e-05 的预测结果============')

        loss_item=0
        num=1
        for epoch in range(cfg.EPOCHS):
            net.train()
            pbar = tqdm(train_loader)
            for i, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item=loss.item()
                pbar.set_postfix({('第'+str(num)+'次loss'): str(loss.item())})
            num+=1
            

                
            if(int(str(epoch)) > 1000  and (loss_item > 0.0004)):
                return 0
                
            if(loss_item < 0.0002):
                path1 = f'./datas/okex/training/'
                torch.save(net.state_dict(), os.path.join(path1,  'training.model.ckpt'))

                
                Datainfo.saveinfo('保存:'+str( os.path.join(path1, 'training.model.ckpt')))
                print('保存:'+str( os.path.join(path1, 'training.model.ckpt')))

                return int(str(epoch))

            if(int(str(epoch))> 30 and loss_item>=0.01):
                return 0

            if(int(str(epoch))> 1000):
    
                return 0



class test_net(nn.Module):

    def __init__(self):
        super(test_net, self).__init__()
        self.fc1 = nn.Linear(36*9, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 256)
        self.fc10 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)
        x = F.relu(x)
        x = self.fc10(x)
        return x

    def gettest(self):


        device = torch.device("cuda")
        train_dataset = Test_CsvDataset(f'./datas/okex/eth/ethclose.csv', 9)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=500,
          shuffle=True,
          num_workers=0)
        train_steps = len(train_loader)
        net = test_net.cuda(self)

        net.load_state_dict(torch.load(f'./datas/okex/training/training.model.ckpt'))
        criterion = nn.MSELoss()
        net.eval()
        for inputs, labels, out_max, time in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            outputs = outputs.cpu() * out_max
            labels = labels.cpu() * out_max
            
            
            isbuy = ''

            if(outputs.item()>labels.numpy()[0][0]):
                isbuy = '买入'
            if(outputs.item()<labels.numpy()[0][0]):
                isbuy = '卖出'
                       

            sendtext = '\n最终loss损失率值: '+str(loss.item())+"\n现在eth-usd-swap的收盘价是："+str(labels.numpy()[0][0])+"\n预测价格： "+str(outputs.item())+"\n是否买入："+isbuy+"\n是否卖出："+isbuy
                
            Datainfo.saveinfo(sendtext)
                    
            Datainfo.save_finalinfo(sendtext)

            #SendDingding.sender(sendtext)

            return isbuy
                


class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return json.JSONEncoder.default(self,obj)


class Datainfo:


    def getdatainfo_full(symbol,symbolmin,minute):

        

        t = time.time()

        #print (t)                       #原始时间数据
        #print (int(t))                  #秒级时间戳
        #print (int(round(t * 1000)))    #毫秒级时间戳
        #print (int(round(t * 1000000))) #微秒级时间戳
        tt = str((int(t * 1000)))
        ttt = str(int(round(t * 1000)))

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
        'referer': 'https://www.okex.com/markets/swap-info/'+symbolmin+'-usd',
        'cookie': 'locale=zh_CN; _gcl_au=1.1.1849415495.'+str(tt)+'; _ga=GA1.2.1506507962.'+str(tt)+'; _gid=GA1.2.256681666.'+str(tt)+'; first_ref=https^%^3A^%^2F^%^2Fwww.okex.com^%^2Fcaptcha^%^3Fto^%^3DaHR0cHM6Ly93d3cub2tleC5jb20vbWFya2V0cy9zd2FwLWRhdGEvZXRoLXVzZA^%^3D^%^3D; _gat_UA-35324627-3=1; amp_56bf9d=gqC_GMDGl4q5Tk-BJhT-oP...1f711b989.1f711fv58.0.2.2',
        }

        params = (
        ('granularity', str(int(minute)*60)),
        ('size', '1000'),
        ('t', str(ttt)),
        )
        response = r.get('https://www.okex.com/v2/perpetual/pc/public/instruments/'+symbol+'/candles', headers=headers, params=params)

        if response.cookies.get_dict(): #保持cookie有效 
                s=r.session()
                c = r.cookies.RequestsCookieJar()#定义一个cookie对象
                c.set('cookie-name', 'cookie-value')#增加cookie的值
                s.cookies.update(c)#更新s的cookie
                s.get(url = 'https://www.okex.com/v2/perpetual/pc/public/instruments/'+symbol+'/candles?granularity=900&size=1000&t='+str(ttt))
        dw = pd.DataFrame(eval(json.dumps(response.json()))['data'])
        #print(dw)
        dw.columns = ['timestamps','open','high','low','close','vol','p']
        datelist = []
        for timestamp in dw['timestamps']:
            datelist.append(timestamp.split('.000Z')[0].replace('T',' '))
        dw['timestamps'] = datelist
        dw['timestamps'] = pd.to_datetime(dw['timestamps'])+pd.to_timedelta('8 hours')
        #df['timestamps'] = df['timestamps'].apply(lambda x:time.mktime(time.strptime(str(x),'%Y-%m-%d %H:%M:%S')))
        #print(dw)
        dw['vol'] = list(map(float, dw['vol'].values))
        #print(symbol)
        dw.to_csv(f'./datas/okex/symbol/'+symbol+'.csv',index = False)
        dw.columns = ['timestamps','open','high','low',symbolmin,'vol','p']
        dw[['timestamps',symbolmin]].to_csv(f'./datas/okex/symbolmin/'+symbolmin+'.csv',index = False)

    #保存所有数据
    def saveallpart():

        path_list = Path(f'./datas/okex/symbolmin/')
        #录入今天的数据并且保存
        df = pd.DataFrame(columns=['timestamps']).set_index('timestamps')
        print(path_list)
        for file in path_list.iterdir():
            df0 = pd.read_csv(file).set_index('timestamps')
            df = df.merge(df0, left_index=True, right_index=True, how='outer')
        df.head()

        df.to_csv(f'./datas/okex/eth/ethclose.csv')
        Datainfo.saveinfo('保存所有close数据完毕...')
        print('保存所有close数据完毕...')

        df = pd.read_csv(f'./datas/okex/eth/ethclose.csv')
        #循环改变第二列数据为eth的值，并且保存          
        df_id = df['eth']
        
        df = df.drop('eth',axis=1)
        df.insert(1,'eth',df_id)
        df.to_csv(f'./datas/okex/eth/ethclose.csv',index=False)

        Datainfo.saveinfo('更换eth的列的close数据完毕...')
        print('更换eth的列的close数据完毕...')

    def isbuy(minute):

        
        Datainfo.saveinfo('开始获取是否可以买入ismarket。。。')

        api_key,secret_key,passphrase,flag = Datainfo.get_userinfo()

        #判断订单是否大于5单，大于则不买入
        #trade api
        tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
        list_string_buy = ['buy']
        list_string_sell = ['sell']
        list_text = list(pd.DataFrame(eval(str(tradeAPI.get_fills()))['data'])['side'].head(100).values)
        all_words_buy = list(filter(lambda text: all([word in text for word in list_string_buy]), list_text ))
        all_words_sell = list(filter(lambda text: all([word in text for word in list_string_sell]), list_text ))
        Datainfo.saveinfo('总计：--->>>'+str(len(all_words_buy) - len(all_words_sell))+' 单 。。。>>>')
        print('总计：--->>>',len(all_words_buy) - len(all_words_sell),' 单 。。。>>>')
        if(len(all_words_buy) - len(all_words_sell)>30):
            Datainfo.saveinfo('买单大于30单返回。。。>>>')
            
            return '30单'

        if(len(all_words_buy) < len(all_words_sell) - 5):
            Datainfo.saveinfo('买单小于卖单返回。。。>>>')
            return '买单小于卖单'

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
            'referer': 'https://www.okex.com/markets/swap-data/eth-usd',
            'cookie': '_gcl_au=1.1.1849415495.'+str(tt)+'; _ga=GA1.2.1506507962.'+str(tt)+'; first_ref=https^%^3A^%^2F^%^2Fwww.okex.com^%^2Fcaptcha^%^3Fto^%^3DaHR0cHM6Ly93d3cub2tleC5jb20vbWFya2V0cy9zd2FwLWRhdGEvZXRoLXVzZA^%^3D^%^3D; locale=zh_CN; _gid=GA1.2.802198982.'+str(tt)+'; amp_56bf9d=gqC_GMDGl4q5Tk-BJhT-oP...1f8fiso4n.1f8fiu841.1.2.3',
        }

        params = (
            ('t', str(ttt)),
            ('unitType', '0'),
        )

        response = r.get('https://www.okex.com/v3/futures/pc/market/takerTradeVolume/ETH', headers=headers, params=params)

        if response.cookies.get_dict(): #保持cookie有效 
                s=r.session()
                c = r.cookies.RequestsCookieJar()#定义一个cookie对象
                c.set('cookie-name', 'cookie-value')#增加cookie的值
                s.cookies.update(c)#更新s的cookie
                s.get(url = 'https://www.okex.com/v3/futures/pc/market/takerTradeVolume/ETH?t='+str(ttt)+'&unitType=0')
        df = pd.DataFrame(response.json()['data'])
        df.to_csv(f'./datas/okex/eth.csv',index=False)
        df = pd.read_csv(f'./datas/okex/eth.csv')
        df['timestamps'] = pd.to_datetime(df['timestamps'],unit='ms')+pd.to_timedelta('8 hours')

        buyVolumes = df['buyVolumes'].tail(20).values
        sellVolumes = df['sellVolumes'].tail(20).values

        print(df)
        print(str(datetime.now())+'--->>>(sum(buyVolumes)/len(buyVolumes)) / (sum(sellVolumes)/len(sellVolumes))的计算结果--->>>',(sum(buyVolumes)/len(buyVolumes)) / (sum(sellVolumes)/len(sellVolumes)))

        f_info = open(f'./datas/symbollist.txt',"r",encoding='utf-8')   #设置文件对象
        symbollist = list(eval(f_info.read()))     #将txt文件的所有内容读入到字符串str中
        f_info = open(f'./datas/symbolminlist.txt',"r",encoding='utf-8')   #设置文件对象
        symbolminlist = list(eval(f_info.read()))     #将txt文件的所有内容读入到字符串str中)
        
        for i in range(10000):
            try:
                i = 0
                #集合所有的close文件

                p1 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[0],symbolminlist[0],5])
                p2 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[1],symbolminlist[1],5])
                p3 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[2],symbolminlist[2],5])
                p4 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[3],symbolminlist[3],5])
                p5 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[4],symbolminlist[4],5])
                p6 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[5],symbolminlist[5],5])
                p7 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[6],symbolminlist[6],5])
                p8 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[7],symbolminlist[7],5])
                p9 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[8],symbolminlist[8],5])
                p10 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[9],symbolminlist[9],5])
                p11 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[10],symbolminlist[10],5])
                p12 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[11],symbolminlist[11],5])
                p13 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[12],symbolminlist[12],5])
                p14 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[13],symbolminlist[13],5])
                p15 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[14],symbolminlist[14],5])
                p16 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[15],symbolminlist[15],5])
                p17 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[16],symbolminlist[16],5])
                p18 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[17],symbolminlist[17],5])
                p19 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[18],symbolminlist[18],5])
                p20 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[19],symbolminlist[19],5])
                p21 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[20],symbolminlist[20],5])
                p22 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[21],symbolminlist[21],5])
                p23 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[22],symbolminlist[22],5])
                p24 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[23],symbolminlist[23],5])
                p25 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[24],symbolminlist[24],5])
                p26 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[25],symbolminlist[25],5])
                p27 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[26],symbolminlist[26],5])
                p28 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[27],symbolminlist[27],5])
                p29 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[28],symbolminlist[28],5])
                p30 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[29],symbolminlist[29],5])
                p31 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[30],symbolminlist[30],5])
                p32 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[31],symbolminlist[31],5])
                p33 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[32],symbolminlist[32],5])
                p34 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[33],symbolminlist[33],5])
                p35 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[34],symbolminlist[34],5])
                p36 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[35],symbolminlist[35],5])
                p37 = multiprocessing.Process(target = Datainfo.getdatainfo_full,args=[symbollist[36],symbolminlist[36],5])
  
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

                time.sleep(1)

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

                time.sleep(1)

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

                time.sleep(1)

                p31.start()
                p32.start()
                p33.start()
                p34.start()
                p35.start()
                p36.start()
                p37.start()
          
                
                
                
                p31.join()
                p32.join()
                p33.join()
                p34.join()
                p35.join()
                p36.join()
                p37.join()

                time.sleep(2)

                dw = pd.read_csv(f'./datas/okex/symbol/ETH-USD-SWAP.csv')
                #获取obv参数
                Datainfo.getfulldata(dw)
                dw = pd.read_csv(f'./datas/okex/symbol/ETH-USD-SWAP.csv')
                #===判断是否买入或者卖出
                print('obv-->>',dw['obv'].tail(1).values ,'MA_obv-->>', dw['maobv'].tail(1).values)

                time.sleep(3)
                #保存所有的close数据
                Datainfo.saveallpart()

                time.sleep(1)

                #人工智能计算结果
                learning = Datainfo.getnextdata()
                print('learning--->>>',learning)

                time.sleep(1)

                if((sum(buyVolumes)/len(buyVolumes)) /(sum(sellVolumes)/len(sellVolumes)) > 1.01 and dw['obv'].tail(1).values > dw['maobv'].tail(1).values and learning == '买入'):
                    print('买入')
                    sendtext = '获取数据完毕。。。   判断为： -->>True-->>>  '+str(minute)+'分钟--->>>buyVolumes-->>'+str(df['buyVolumes'].values[-1])+'--->>>sellVolumes-->>'+str(df['sellVolumes'].values[-1])+'  ,预测结果-->>正确   -->>我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫 ！^_^'
                    Datainfo.saveinfo(sendtext)
                    return '买入'
                elif((sum(buyVolumes)/len(buyVolumes)) /(sum(sellVolumes)/len(sellVolumes)) < 0.99 and dw['obv'].tail(1).values < dw['maobv'].tail(1).values and learning == '卖出'):
                    print('卖出')
                    sendtext = '获取数据完毕。。。   判断为： -->>True-->>>  '+str(minute)+'分钟--->>>buyVolumes-->>'+str(df['buyVolumes'].values[-1])+'--->>>sellVolumes-->>'+str(df['sellVolumes'].values[-1])+'  ,预测结果-->>正确   -->>我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫 ！^_^'
                    Datainfo.saveinfo(sendtext)
                    return '卖出'
                else:
                    print('不买卖')
                    return '不买卖'
                break
            except:
                time.sleep(3)
                continue


    def getfulldata(df):


        #获取参数历史数据
        df['will'] = ta.WILLR(df['high'].values,df['low'].values,df['close'].values,timeperiod=14)
        df['upper'], df['middle'], df['lower'] = ta.BBANDS(
                        df.close.values,
                        timeperiod=20,
                        # number of non-biased standard deviations from the mean
                        nbdevup=2,
                        nbdevdn=2,
                        # Moving average type: simple moving average here
                        matype=0)
        df["rsi"] = ta.RSI(df['close'], timeperiod=14)
        df['slowk'], df['slowd'] = ta.STOCH(df['high'].values,
                                df['low'].values,
                                df['close'].values,
                                fastk_period=9,
                                slowk_period=3,
                                slowk_matype=0,
                                slowd_period=3,
                                slowd_matype=0)
        df["DEMA"] = ta.DEMA(df['close'].values, timeperiod=30)
        # MA - Moving average 移动平均线
        # 函数名：MA
        # 名称： 移动平均线
        # 简介：移动平均线，Moving Average，简称MA，原本的意思是移动平均，由于我们将其制作成线形，所以一般称之为移动平均线，简称均线。它是将某一段时间的收盘价之和除以该周期。 比如日线MA5指5天内的收盘价除以5 。
        # real = MA(close, timeperiod=30, matype=0)
        df["MA"] = ta.MA(df['close'].values, timeperiod=30, matype=0)
        # EMA和MACD
        df['obv'] = ta.OBV(df['close'].values,df['vol'].values)
        df['maobv'] = ta.MA(df['obv'].values, timeperiod=30, matype=0)
        df["MACD_macd"],df["MACD_macdsignal"],df["MACD_macdhist"] = ta.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=60)
        df['macd'] = 2*(df["MACD_macd"]-df["MACD_macdsignal"])
        df['upper'], df['middle'], df['lower'] = ta.BBANDS(
                    df.close.values,
                    timeperiod=26,
                    # number of non-biased standard deviations from the mean
                    nbdevup=2,
                    nbdevdn=2,
                    # Moving average type: simple moving average here
                    matype=0)
        df.to_csv(f'./datas/okex/symbol/ETH-USD-SWAP.csv',index = False)


    #获取下个预期数值的方法
    def getnextdata():

        #循环开始训练并且匹配结果计算运行
        for i in range(1000000) :
     
            train_list = 0
            train = train_net()
            train_list = train.gettrain()
            #se.sender("第%8s轮训练完毕。。。"%(i+1))
            if(train_list == 0):
                continue
            else:
                break

        testnet = test_net()
        result = testnet.gettest()
        return result
        

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
    def orderbuy(api_key, secret_key, passphrase, flag):

        # account api
        accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
        
        # 设置持仓模式  Set Position mode
        result = accountAPI.get_position_mode('long_short_mode')
        # 设置杠杆倍数  Set Leverage
        result = accountAPI.set_leverage(instId='ETH-USD-SWAP', lever='100', mgnMode='cross')
        #Datainfo.saveinfo('设置100倍保证金杠杆完毕。。。')
        # trade api
        tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
        # 批量下单  Place Multiple Orders
        # 批量下单  Place Multiple Orders
        result = tradeAPI.place_multiple_orders([
             {'instId': 'ETH-USD-SWAP', 'tdMode': 'cross', 'side': 'buy', 'ordType': 'market', 'sz': '1',
              'posSide': 'long',
              'clOrdId': 'a12344', 'tag': 'test1210'},
    

         ])
        print(result)

        #Datainfo.saveinfo('下单完毕。。。')

        lastprice = Datainfo.getlastprice(api_key, secret_key, passphrase, flag)

        #Datainfo.saveinfo('获取最新价格。。。'+str(lastprice))
        
        # 调整保证金  Increase/Decrease margint
        result = accountAPI.Adjustment_margin('ETH-USD-SWAP', 'short', 'add', '5')
        #Datainfo.saveinfo('调整保证金完毕。。。')

        # 策略委托下单  Place Algo Order
        result = tradeAPI.place_algo_order('ETH-USD-SWAP', 'cross', 'sell', ordType='conditional',
                                            sz='1',posSide='long', tpTriggerPx=str(float(lastprice)+50), tpOrdPx=str(float(lastprice)+50))
        #Datainfo.saveinfo(str(datetime.now())+'设置止盈完毕。。。'+str(float(lastprice)+50))


        sendtext = str(datetime.now())+'--->>>100倍杠杆，全仓委托：买入ETH-USD-SWAP -->> 1笔，价格是'+str(lastprice)+'，设置止盈完毕。。。'+str(float(lastprice)+50)
        Datainfo.save_finalinfo(str(datetime.now())+'--->>>我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫 ！^_^     -->>'+sendtext)
        SendDingding.sender(sendtext)

    #设置自动下单
    def ordersell(api_key, secret_key, passphrase, flag):

        # account api
        accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
        
        # 设置持仓模式  Set Position mode
        result = accountAPI.get_position_mode('long_short_mode')
        # 设置杠杆倍数  Set Leverage
        result = accountAPI.set_leverage(instId='ETH-USD-SWAP', lever='100', mgnMode='cross')
        #Datainfo.saveinfo('设置100倍保证金杠杆完毕。。。')
        # trade api
        tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
        # 批量下单  Place Multiple Orders
        result = tradeAPI.place_order(instId='ETH-USD-SWAP', tdMode='cross', side='sell', posSide='short',
                              ordType='market', sz='1')
        print(result)

        #Datainfo.saveinfo('下单完毕。。。')

        lastprice = Datainfo.getlastprice(api_key, secret_key, passphrase, flag)

        #Datainfo.saveinfo('获取最新价格。。。'+str(lastprice))
        
        # 调整保证金  Increase/Decrease margint
        result = accountAPI.Adjustment_margin('ETH-USD-SWAP', 'short', 'add', '5')
        #Datainfo.saveinfo('调整保证金完毕。。。')

        # 策略委托下单  Place Algo Order
        result = tradeAPI.place_algo_order('ETH-USD-SWAP', 'cross', 'buy', ordType='conditional',
                                            sz='1',posSide='short', tpTriggerPx=str(float(lastprice)-50), tpOrdPx=str(float(lastprice)-50))
        #Datainfo.saveinfo(str(datetime.now)+'设置止盈完毕。。。'+str(float(lastprice)-50))


        sendtext = str(datetime.now())+'--->>>卖出100倍杠杆，全仓委托：ETH-USD-SWAP -->> 1笔，价格是'+str(lastprice)+'，设置止盈完毕。。。'+str(float(lastprice)-50)
        Datainfo.save_finalinfo(str(datetime.now())+'--->>>我们是守护者，也是一群时刻对抗危险和疯狂的可怜虫 ！^_^     -->>'+sendtext)
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


            #6个进程开始运行
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
            day_buy = f_day.read()[-400:]     #将txt文件的所有内容读入到字符串str中
            f_day.close()   #将文件关闭
            if(day_buy):
                self.textBrowsertwo.clear()
                self.textBrowsertwo.append(day_buy)

            f_info = open(f'./datas/log/infodata.txt',"r",encoding='utf-8')   #设置文件对象
            infodata = f_info.read()[-400:]     #将txt文件的所有内容读入到字符串str中
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
            #self.getdatainfo('5')
        
        def getdatainfo(self,minute):

            time.sleep(8)
            
            print(minute)
            isbuy  =  Datainfo.isbuy(minute)

            if('15单' == isbuy or '买单小于卖单'== isbuy):
                Datainfo.saveinfo('预测不买入。。。')
                return
            elif('买入' == isbuy):
                api_key, secret_key, passphrase, flag = Datainfo.get_userinfo()
                Datainfo.orderbuy(api_key, secret_key, passphrase, flag)
            elif('卖出' == isbuy):
                api_key, secret_key, passphrase, flag = Datainfo.get_userinfo()
                Datainfo.ordersell(api_key, secret_key, passphrase, flag)
            elif('不买卖' == isbuy):
                Datainfo.saveinfo('预测不买卖。。。')
                Datainfo.save_finalinfo('预测不买卖。。。')




           
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