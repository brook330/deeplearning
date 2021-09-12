# -*- coding: gbk -*-
import okex.Account_api as Account
import okex.Funding_api as Funding
import okex.Market_api as Market
import okex.Public_api as Public
import okex.Trade_api as Trade
import okex.subAccount_api as SubAccount
import okex.status_api as Status
from btcbuy import Datainfo
import datetime
import pandas as pd

api_key,secret_key,passphrase,flag = Datainfo.get_userinfo()

#判断订单是否大于5单，大于则不买入
#trade api
tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
list_string_buy = ['buy']
list_string_sell = ['sell']
print(tradeAPI.get_fills())
list_text = list(pd.DataFrame(eval(str(tradeAPI.get_fills()))['data'])['side'].head(300).values)
all_words_buy = list(filter(lambda text: all([word in text for word in list_string_buy]), list_text ))
all_words_sell = list(filter(lambda text: all([word in text for word in list_string_sell]), list_text ))
 
print(len(all_words_buy),len(all_words_sell))