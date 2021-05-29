# -*- coding: utf-8 -*-
import okex.Account_api as Account
import okex.Funding_api as Funding
import okex.Market_api as Market
import okex.Public_api as Public
import okex.Trade_api as Trade
import okex.subAccount_api as SubAccount
import okex.status_api as Status
import json
import requests as r
import pandas as pd
import datetime,time
from datetime import datetime
import numpy 



if __name__ == '__main__':
    
    #with open('api.json', 'r', encoding='utf-8') as f:
    #    obj = json.loads(f.read())


    #api_key = obj['api_key']
    #secret_key = obj['secret_key']
    #passphrase = obj['passphrase']

    ## flag是实盘与模拟盘的切换参数 flag is the key parameter which can help you to change between demo and real trading.
    ## flag = '1'  # 模拟盘 demo trading
    #flag = '0'  # 实盘 real trading

    ## account api
    #accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
    ## 查看账户持仓风险 GET Position_risk
    #result = accountAPI.get_position_risk('SWAP')
    #print(result)
    ## 设置持仓模式  Set Position mode
    #result = accountAPI.get_position_mode('long_short_mode')
    ## 设置杠杆倍数  Set Leverage
    #result = accountAPI.set_leverage(instId='ETH-USD-SWAP', lever='100', mgnMode='cross')
    ## trade api
    #tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
    ### 批量下单  Place Multiple Orders
    ##result = tradeAPI.place_multiple_orders([
    ##     {'instId': 'ETH-USD-SWAP', 'tdMode': 'cross', 'side': 'buy', 'ordType': 'market', 'sz': '1',
    ##      'posSide': 'long',
    ##      'clOrdId': 'a12344', 'tag': 'test1210'},

    ## ])
    ##print(result)


    ### 策略委托下单  Place Algo Order
    ##result = tradeAPI.place_algo_order('ETH-USD-SWAP', 'cross', 'sell', ordType='conditional',
    ##                                    sz='1',posSide='long', tpTriggerPx='1990', tpOrdPx='1989')

    ## 调整保证金  Increase/Decrease margint
    #result = accountAPI.Adjustment_margin('ETH-USD-SWAP', 'short', 'add', '1')

    t = time.time()

    #print (t)                       #原始时间数据
    #print (int(t))                  #秒级时间戳
    #print (int(round(t * 1000)))    #毫秒级时间戳
    #print (int(round(t * 1000000))) #微秒级时间戳
    tt = str((int(t * 1000)))
    ttt = str(int(round(t * 1000)))
    
    headers = {
        'authority': 'www.binancezh.com',
        'x-trace-id': 'a08453b5-01dc-4660-9db1-c4ab1c7082e6',
        'csrftoken': 'd41d8cd98f00b204e9800998ecf8427e',
        'x-ui-request-trace': 'a08453b5-01dc-4660-9db1-c4ab1c7082e6',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36',
        'content-type': 'application/json',
        'lang': 'zh-CN',
        'fvideo-id': '31fc1e85e2aba47da19d61a18fbad3c1658d6cbf',
        'sec-ch-ua-mobile': '?0',
        'device-info': 'eyJzY3JlZW5fcmVzb2x1dGlvbiI6IjE5MjAsMTA4MCIsImF2YWlsYWJsZV9zY3JlZW5fcmVzb2x1dGlvbiI6IjE5MjAsMTA0MCIsInN5c3RlbV92ZXJzaW9uIjoiV2luZG93cyAxMCIsImJyYW5kX21vZGVsIjoidW5rbm93biIsInN5c3RlbV9sYW5nIjoiemgtQ04iLCJ0aW1lem9uZSI6IkdNVCs4IiwidGltZXpvbmVPZmZzZXQiOi00ODAsInVzZXJfYWdlbnQiOiJNb3ppbGxhLzUuMCAoV2luZG93cyBOVCAxMC4wOyBXaW42NDsgeDY0KSBBcHBsZVdlYktpdC81MzcuMzYgKEtIVE1MLCBsaWtlIEdlY2tvKSBDaHJvbWUvOTAuMC40NDMwLjkzIFNhZmFyaS81MzcuMzYiLCJsaXN0X3BsdWdpbiI6IkNocm9tZSBQREYgUGx1Z2luLENocm9tZSBQREYgVmlld2VyLE5hdGl2ZSBDbGllbnQiLCJjYW52YXNfY29kZSI6Ijc4OTczMzhkIiwid2ViZ2xfdmVuZG9yIjoiR29vZ2xlIEluYy4gKEFNRCkiLCJ3ZWJnbF9yZW5kZXJlciI6IkFOR0xFIChBTUQsIEFNRCBSYWRlb24gUlggNTgwIDIwNDhTUCBEaXJlY3QzRDExIHZzXzVfMCBwc181XzAsIEQzRDExLTI3LjIwLjIxMDAyLjExMikiLCJhdWRpbyI6IjEyNC4wNDM0NzUyNzUxNjA3NCIsInBsYXRmb3JtIjoiV2luMzIiLCJ3ZWJfdGltZXpvbmUiOiJBc2lhL1NoYW5naGFpIiwiZGV2aWNlX25hbWUiOiJDaHJvbWUgVjkwLjAuNDQzMC45MyAoV2luZG93cykiLCJmaW5nZXJwcmludCI6IjJmZjk0MGM1NDgzNzZkMzI1YWQwNTk4MGVhNTg3MzU1IiwiZGV2aWNlX2lkIjoiIiwicmVsYXRlZF9kZXZpY2VfaWRzIjoiIn0=',
        'bnc-uuid': 'e63ec6b9-0927-475c-98e1-4de1a573a76c',
        'clienttype': 'web',
        'sec-ch-ua': '^\\^',
        'accept': '*/*',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
        'referer': 'https://www.binancezh.sh/zh-CN/futures/funding-history/quarterly/4',
        'accept-language': 'zh-CN,zh;q=0.9',
        'cookie': 'cid=TI6wFBRw; _ga=GA1.2.1908414457.'+tt+'; _gid=GA1.2.1560338037.'+tt+'; bnc-uuid=e63ec6b9-0927-475c-98e1-4de1a573a76c; sajssdk_2015_cross_new_user=1; sensorsdata2015jssdkcross=^%^7B^%^22distinct_id^%^22^%^3A^%^22179b4d85052b9e-0020925ad17fc1-d7e1739-2073600-179b4d85053c26^%^22^%^2C^%^22first_id^%^22^%^3A^%^22^%^22^%^2C^%^22props^%^22^%^3A^%^7B^%^22^%^24latest_traffic_source_type^%^22^%^3A^%^22^%^E7^%^9B^%^B4^%^E6^%^8E^%^A5^%^E6^%^B5^%^81^%^E9^%^87^%^8F^%^22^%^2C^%^22^%^24latest_search_keyword^%^22^%^3A^%^22^%^E6^%^9C^%^AA^%^E5^%^8F^%^96^%^E5^%^88^%^B0^%^E5^%^80^%^BC_^%^E7^%^9B^%^B4^%^E6^%^8E^%^A5^%^E6^%^89^%^93^%^E5^%^BC^%^80^%^22^%^2C^%^22^%^24latest_referrer^%^22^%^3A^%^22^%^22^%^7D^%^2C^%^22^%^24device_id^%^22^%^3A^%^22179b4d85052b9e-0020925ad17fc1-d7e1739-2073600-179b4d85053c26^%^22^%^7D; userPreferredCurrency=USD_USD; BNC_FV_KEY=31fc1e85e2aba47da19d61a18fbad3c1658d6cbf; BNC_FV_KEY_EXPIRE='+tt+'; aliyungf_tc=4a44a0530032a5bf23636c1a228da72affdf5652443ba5adc9281e68087154c2; monitor-uuid=ac1f54a4-7ca1-4102-b029-df9b0e6d118c; lang=zh-cn',
    }

    params = (
        ('pair', 'ETHUSD'),
        ('period', '5m'),
        ('contractType', 'PERPETUAL'),
    )

    response = r.get('https://www.binancezh.com/bapi/futures/v1/public/delivery/data/takerBuySellVol', headers=headers, params=params)

    if response.cookies.get_dict(): #保持cookie有效 
            s=r.session()
            c = r.cookies.RequestsCookieJar()#定义一个cookie对象
            c.set('cookie-name', 'cookie-value')#增加cookie的值
            s.cookies.update(c)#更新s的cookie
            s.get(url = 'https://www.binancezh.sh/bapi/futures/v1/public/delivery/data/takerBuySellVol?pair=ETHUSD&period=5m&contractType=PERPETUAL')

    df = pd.DataFrame(eval(json.dumps(response.json())))
    datelist = []
    for timestamp in df['timestamp']:
        datelist.append(datetime.fromtimestamp(timestamp/1000).strftime("%Y-%m-%d %H:%M:%S"))

    df['timestamp'] = pd.to_datetime(datelist)
    df['plot']=numpy.divide(df['takerBuyVolValue'].values.astype(numpy.float64),df['takerSellVolValue'].values.astype(numpy.float64))
    df = df[['timestamp','pair','plot','takerBuyVolValue','takerBuyVol','takerSellVol','takerSellVolValue']]
    df.to_csv('ethusd.csv',index = False)
    df = pd.read_csv('ethusd.csv')
    print(df['takerBuyVolValue'].values[-1]>df['takerBuyVolValue'].values[-2] and df['takerBuyVol'].values[-1]>df['takerBuyVol'].values[-2] and df['plot'].values[-1]>1 and df['plot'].values[-1] > df['plot'].values[-2])
