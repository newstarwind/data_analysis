'''
根据Bill提供的SPY，QQQ，DIA等回测图形，推敲基于30 min bar的日内交易策略。
'''

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from scipy import stats
import os
figsize = (10, 8)
plt.style.use(['seaborn'])
# %% Prepare all data
df = pd.read_csv('data' + os.sep + 'spy.csv', parse_dates=True,
                 index_col=0).dropna().sort_index(ascending=True).round(2)

df = df.resample('30T', closed='right', label='right').last().dropna()
spy = df.copy()
spy.index = [x.strftime('%Y-%m-%d %H:%M') for x in df.index]
spy['fwd returns'] = (spy['price'].shift(-5) - spy['price']
                      ) / spy['price']  # returns after 5 bars
spy['ema8'] = talib.EMA(spy['price'].values, timeperiod=8)
spy['ema21'] = talib.EMA(spy['price'].values, timeperiod=21)
spy['delta'] = spy['ema8'] - spy['ema21']
spy = spy.dropna()
spy['price'].plot(figsize=figsize)
spy['ema21'].plot()
spy['ema8'].plot()
# %%
print '''
我们先看一看cornor RSI
正在思考如何进行矢量运算来实现
'''
# %%
def rsi(prices, window):
    rsi = talib.RSI(prices, window)
    return rsi

def duration(prices):
    duration = []
    for i in range(len(prices)):
        if i >= 1:
            if prices[i] > prices[i - 1]:
                if len(duration) == 0:
                    duration.append(1.0)
                elif duration[-1] >= 0:
                    duration.append(duration[-1] + 1.0)
                elif duration[-1] < 0:
                    duration.append(1.0)
            if prices[i] == prices[i - 1]:
                duration.append(0.0)
            if prices[i] < prices[i-1]:
                if len(duration) == 0:
                    duration.append(-1.0)
                elif duration[-1] >= 0:
                    duration.append(-1.0)
                elif duration[-1] < 0:
                    duration.append(duration[-1] - 1.0)

    return duration

def rank(prices):
    data = pd.Series(prices).pct_change()
    current = data.values[-1]
    return 100 * sum(data < current)/float(len(prices.values))

def connor_rsi(prices, rsi_window = 3, steak_window = 2, rank_window = 100 ):
    normal_rsi = rsi(np.array(prices), rsi_window)
    steak_rsi  = rsi(np.array(duration(prices)),steak_window)
    per_rank   = rank(prices[-rank_window:])
    print normal_rsi[-1], steak_rsi[-1], per_rank
    return (normal_rsi[-1] + steak_rsi[-1] + per_rank) / 3.0

print connor_rsi(spy.price)

# %%
def steak_rsi(df, name, window):
    '''
    voctor based calculation of Steak RSI
    '''    
    temp = pd.DataFrame(index= df.index, columns=['pct_change', 
                                                   'duration', 
                                                   'steak', 
                                                   'steak_rsi'])
    temp['pct_change'] = pd.Series(df[name]).pct_change()
    temp['pct_change'] = temp['pct_change'].fillna(0)
    temp.loc[temp['pct_change']>0,  'duration'] =  1.0
    temp.loc[temp['pct_change']<0,  'duration'] = -1.0
    temp.loc[temp['pct_change']==0, 'duration'] =  0.0
    result = pd.DataFrame(index = temp.index, columns = ['steak', 'steak_rsi'])
    for i, (index, row) in enumerate(temp.iterrows()):
        current = row['duration']
        last    = temp.iloc[i-1]['duration']
        result_last = result.iloc[i-1]['steak']
        if current == 0:
            result.iloc[i]['steak'] = 0.0
        elif current > 0:
            if last <= 0:
                result.iloc[i]['steak'] = 1.0
            elif last > 0:
                result.iloc[i]['steak'] = result_last + 1.0
        elif current < 0:
            if last >= 0:
                result.iloc[i]['steak'] = -1.0
            elif last < 0:
                result.iloc[i]['steak'] = result_last -1.0

    temp['steak'] = [ float(x) for x in result['steak']]
    temp['steak_rsi'] = talib.RSI(temp['steak'].values,window)
    return temp['steak_rsi']
        
print steak_rsi(spy, 'price', 2)

# %%
def rank(df, name, window):
    data = pd.Series(prices).pct_change()
    current = data.values[-1]
    return 100 * sum(data < current)/float(len(prices.values))

def rank(df, name, window):
    temp = pd.DataFrame(index= df.index, columns=['pct_change', 
                                                   'rank'])
    temp['pct_change'] = df[name].pct_change()
    temp['rank'] = sum(df[name].rolling(window) < df[name].rolling(1).sum()) / float(window)                                               
    return 100 * temp['rank']

name = 'price'
window = 100
temp = pd.DataFrame(index= df.index, columns=['pct_change', 'rank'])
temp['pct_change'] = df[name].pct_change()
# print df[name].rolling(window) < df[name].rolling(1).sum()
print df[name].rolling(window) 

# print rank(spy,'price',100)

# %%
print '''
我们考察一下如果以Rolling Zscore作为入场条件，是否合适？ 先设置window = 21
'''

# %%
def zscore(x, window):
    '''
    x： 序列
    window： rolling 宽度, 越大愈能过滤杂波
    计算rolling zscore，并且归一化到 [-1 : +1] 之间。
    '''
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    min = np.min(z)
    max = np.max(z)
    z = (z - min) / (max - min)
    z = z * 2 - 1
    return z


spy['zscore'] = zscore(spy['delta'], window=36)
spy['zscore'].plot(figsize=figsize)
plt.legend()

# %%
spy = spy.dropna()
print spy['zscore'].describe()
bottom = np.percentile(spy['zscore'], 20)
high = np.percentile(spy['zscore'], 80)
print bottom, high
# %%
plt.figure(figsize=figsize)
worst_days = spy['zscore'] < bottom
spy['price'].plot()
spy.loc[worst_days, 'price'].plot()
plt.show()
# %%
plt.figure(figsize=figsize)
spy['fwd returns'].groupby(pd.qcut(spy['zscore'], 10)).mean().plot(kind='bar')


#%%
