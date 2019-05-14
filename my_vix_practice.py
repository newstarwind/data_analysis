'''
study relations between vix and spy
'''

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
figsize = (10,8)

# %%
df = pd.read_csv('data' + os.sep + 'VIX_Study.csv',
                parse_dates=True,
                index_col=0).sort_index(ascending=True).round(2)

df.describe()
#%%

# 前21天年化波动率
# 后21天年化波动率
df['SPY return'] = df['SPY Close'].pct_change() # 当天收盘价变化
df['VIX return'] = df['VIX'].pct_change() # 当天收盘价变化
df['fwd return'] = df['SPY Open'].pct_change().shift(-2) # 第三天开盘价 / 第二天开盘价 - 1
df['hist vol']   = df['SPY return'].rolling(21).std() * np.sqrt(252) * 100
df['fwd vol']   = df['SPY return'].shift(-21).rolling(21).std() * np.sqrt(252)*100

df = df.dropna()
vols = df[['VIX','hist vol','fwd vol']]
vols.corr() # We can see fwd vol has higher relation with VIX, not hist vol
df[['SPY return','VIX return']].corr() # You can see vix beta is about -8

#%%
sns.pairplot(vols)

#%%
'''
先看一看VIX和fwd vol的价差。
* 如果这个价差为正的，表示投资者期待的vol大于实际的vol。投资者愿意出更多的钱买保险。
* 如果这个价差为负的，表示市场的风险比预期的更大。
'''
ex = df['VIX']-df['fwd vol']
ex.hist(bins=100)
plt.title('implied vol - realized vol')
plt.xlabel('%Difference')
plt.ylabel('Occurance')

#%%
'''
价差是大部分正的，说明大部分时间期权以premium交易。
但是在左侧有很长的肥尾，说明有时候投资者大大低估了市场的波动率，比如黑天鹅事件。

下面我们绘出fwd vol被低估的极端时刻，

如果高亮这些波动率被低估的时期，我们发现波动率被低估的时期绝大部分在于Dot Com泡沫破裂
和全球金融危机期间。

可以看到这些低估主要发生在大的危机的起始阶段, 
表示人们还没有意识到危机的严重性.
'''
plt.figure(figsize=(20, 16))
bottom_ex = np.percentile(ex,2.5)
worst_days = ex < bottom_ex
df['SPY Close'].plot()
df.loc[worst_days, 'SPY Close'].plot(style='ro')

#%%
'''
# Excess VIX evaluation 预估的vix与历史vix的差值
VIX - hist vol 我们称之为：Excess VIX, 简称EV。
归一化的EX我们在后面会看到计算方法, 先看一看未归一化的EX.

由于fwd vol在实际中是得不到的，现在我们看一看hist vol 和 implied vol的关系。
这两个指标在实际操作中可以得到， 所以特别重要。

正值表示投资者认为未来的波动率大于过去的波动率, 反之亦然.
和前面一样, implied vol通常是大于historical vol的. 

'''
ev = df['VIX'] - df['hist vol']
ev.hist(bins=100)
print np.percentile(ev,99)
print np.percentile(ev,1)


#%%
'''
但是讨厌的肥尾又出现了!
可以看到这些极端情况发生在市场的底部, 

# 表示在底部人们倾向于高估未来的波动率.
# 反过来说，如果EV足够大，说明大盘到达底部了 !
# 此时，其他做空波动率策略就可以开始启动了。
'''
plt.figure(figsize=(20,16))
ev_high = np.percentile(ev,99)
ev_low = np.percentile(ev,1)
ev_high_days = ev >= ev_high
ev_low_days = ev <= ev_low
df['SPY Close'].plot()
df.loc[ev_high_days, 'SPY Close'].plot(style='ro')
df.loc[ev_low_days, 'SPY Close'].plot(style='go')

#%%
'''
# 我们再来看一看VIX对于SPY波动率的预测成功度.

VIX表示未来的年化波动率, 如果基于自然分布(这是不对的):
* 21天后 price在68%的情况落在正负一个标准差之内.
* 21天后 price在95%的情况落在正负两个个标准差之内.

我们看一看在2008年和2017年这两个极端年份的情况.
# 未来价格边界的计算可以预测21天后的SPY价格上下边界。 
# 对于 upper 来说，1.25 倍已经非常安全的Sell coverd Call.
* sell 1.25 upper, 近10年只有2.74%的失败率
* sell 1.75 upper, 近10年只有0.4%的失败率

# 对于 lower 来说, 2.5 倍才是可以考虑的边界, 但是仍然不应该这样做, 性价比不高.
* sell 2.5 lower put, 2.5 的失败率是0.22%, 但是其每一次价差较大, 无法反脆弱.
'''
# 预测的上下一个标准差的边界，理论上有68%的成功率
multi = 1.25
df['upper'] = (1 + multi * df['VIX'].shift(21) * np.sqrt(21) / np.sqrt(252) / 100) * df['SPY Close'].shift(21)
df['lower'] = (1 - multi * df['VIX'].shift(21) * np.sqrt(21) / np.sqrt(252) / 100) * df['SPY Close'].shift(21)
df.loc['2017', ['SPY Close','upper','lower']].plot(style=['b-', 'g:', 'r:'])

#%%
# lower实际预测成功率
wrong_days = df['lower'] > df['SPY Close']
wrong_num = df.loc[wrong_days,'SPY Close'].count()
total = df['SPY Close'].count()
print '%s percent wrong when spy is lowerer than lower bundary' % np.round(100 * wrong_num / float(total),2)

# upper实际预测成功率
wrong_days = df['upper'] < df['SPY Close']
wrong_num = df.loc[worst_days,'SPY Close'].count()
print '%s percent wrong when spy is higher than upper bundary' % np.round(100 * wrong_num / float(total),2)

#%%
'''
# VS Expected

下面我们看一看过去n天实际变化幅度和预测变化幅度的比较，我们称之为 vs expected。

# vs expected 可以在当天计算出，具有参考意义。
# n = 5，vs expected = 一周以来大盘实际变化幅度 / 一周前预测的变化幅度

可以看出，在大部分时间内，vs expected在 [-1,1]之内，
不管是上涨还是下跌，人们的预测总是能够cover实际变化，
但是， 左侧的肥尾仍然指出，在有些情况下，人们对下跌预测不足。

随着n值加大，预测的准确性进一步加大，n= [10，15]的时候，准确性达到了峰值 84.7%

'''
n = 5
real_change   = df['SPY Close'] - df['SPY Close'].shift(n)
expect_change = df['VIX'].shift(n) / 100 * np.sqrt(n) / np.sqrt(252) * df['SPY Close'].shift(n)

df['expected_change'] = real_change/expect_change  # 实际价格变化幅度 / 预测价格变化幅度
df['expected_change'].hist(bins = 100)

print df.loc[df['expected_change']>0, 'expected_change'].mean() # 大盘上涨
print df.loc[df['expected_change']<0, 'expected_change'].mean() # 大盘下跌

upper_correct = df['expected_change'] <  1
down_correct  = df['expected_change'] > -1
expected_ok   = upper_correct & down_correct
expected_ok_percent = df.loc[expected_ok,'expected_change'].count() / float(df['expected_change'].count()) * 100

print 'correct percent is %s' % expected_ok_percent




#%%


#%%
