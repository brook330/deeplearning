# -*- coding: utf-8 -*-
import requests 
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
from PIL import ImageGrab
import win32api#先要安装pywin32，pip install pywin32

class Senddingding:

    def getAccess_token():
        url = 'https://oapi.dingtalk.com/gettoken?appkey=dingoeih3p2dxwd2qbec&appsecret=ErYLtCvBgXX46Ny0MnSZPR7iR-mLHhWA6_EZzTq_p_BaYord7zDTB2EyHsOO81tu'
        appkey = 'dingoeih3p2dxwd2qbec' # 管理员账号登录开发者平台，应用开发-创建应用-查看详情-appkey
        appsecret = 'ErYLtCvBgXX46Ny0MnSZPR7iR-mLHhWA6_EZzTq_p_BaYord7zDTB2EyHsOO81tu' # 应用里的appsecret
        headers = {
            'Content-Type': "application/x-www-form-urlencoded"
        }
        data = {'appkey': appkey,
                'appsecret': appsecret}
        r = requests.request('GET', url, data=data, headers=headers)
        access_token = r.json()["access_token"]
        return access_token


    def getMedia_id():
        access_token = Senddingding.getAccess_token()  # 拿到接口凭证
        path = 'picture.jpg'  # 文件路径
        url = 'https://oapi.dingtalk.com/media/upload?access_token=' + access_token + '&type=image'
        files = {'media': open(path, 'rb')}
        data = {'access_token': access_token,
                'type': 'image'}
        response = requests.post(url, files=files, data=data)
        json = response.json()
        return json["media_id"]


    def sendImage():
        access_token = Senddingding.getAccess_token()
        media_id = Senddingding.getMedia_id()
        chatid = 'chatf600fe9e823d2eab5ae485d35a74e345'  # 通过jsapi工具获取的群聊id
        url = 'https://oapi.dingtalk.com/chat/send?access_token=' + access_token
        header = {
            'Content-Type': 'application/json'
        }
        data = {'access_token': access_token,
                'chatid': 'chatf600fe9e823d2eab5ae485d35a74e345',
                'msg': {
                    'msgtype': 'image',
                    'image': {'media_id': media_id}
                }}
        r = requests.request('POST', url, data=json.dumps(data), headers=header)
        print(r.json())

class RunSaveImage:

    def  saveimage(self):

        win32api.ShellExecute(0, 'open', r'C:\Users\亏成首富\AppData\Local\Programs\landbridge-desktop\爱交易.exe', '','',1)

        time.sleep(1)

        pic = ImageGrab.grab()
        pic.save("picture.jpg")
        Senddingding.sendImage()

class RuntimeScheduler:


    #5分钟定时截图

    def get_final_image_5M_job(self):

        
        save = RunSaveImage()
        scheduler = BlockingScheduler()
        scheduler.add_job(save.saveimage,'cron', minute='*/5')
        print(scheduler.get_jobs())
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.shutdown()

if __name__ == '__main__':

    dingzhi =  RuntimeScheduler()
    dingzhi.get_final_image_5M_job()