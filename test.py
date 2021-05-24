# -*- coding: utf-8 -*-
import okex.Account_api as Account
import okex.Funding_api as Funding
import okex.Market_api as Market
import okex.Public_api as Public
import okex.Trade_api as Trade
import okex.subAccount_api as SubAccount
import okex.status_api as Status
import json


class Test():

    get

if __name__ == '__main__':
    
    with open('api.json', 'r', encoding='utf-8') as f:
        obj = json.loads(f.read())


    api_key = obj['api_key']
    secret_key = obj['secret_key']
    passphrase = obj['passphrase']

    # flag是实盘与模拟盘的切换参数 flag is the key parameter which can help you to change between demo and real trading.
    # flag = '1'  # 模拟盘 demo trading
    flag = '0'  # 实盘 real trading

    # account api
    accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
    # 查看账户持仓风险 GET Position_risk
    result = accountAPI.get_position_risk('SWAP')
    print(result)
    # 设置持仓模式  Set Position mode
    result = accountAPI.get_position_mode('long_short_mode')
    # 设置杠杆倍数  Set Leverage
    result = accountAPI.set_leverage(instId='ETH-USD-SWAP', lever='100', mgnMode='cross')
    # trade api
    tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
    ## 批量下单  Place Multiple Orders
    #result = tradeAPI.place_multiple_orders([
    #     {'instId': 'ETH-USD-SWAP', 'tdMode': 'cross', 'side': 'buy', 'ordType': 'market', 'sz': '1',
    #      'posSide': 'long',
    #      'clOrdId': 'a12344', 'tag': 'test1210'},

    # ])
    #print(result)


    ## 策略委托下单  Place Algo Order
    #result = tradeAPI.place_algo_order('ETH-USD-SWAP', 'cross', 'sell', ordType='conditional',
    #                                    sz='1',posSide='long', tpTriggerPx='1990', tpOrdPx='1989')

    # 调整保证金  Increase/Decrease margint
    result = accountAPI.Adjustment_margin('ETH-USD-SWAP', 'short', 'add', '1')
