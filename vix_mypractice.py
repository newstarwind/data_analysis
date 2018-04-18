'''
研究VIX和SPY之间的关系
'''
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %%
print '''
从本研究我们可以发现:
# EX : VIX - hist vol：Excess VIX, 简称EV。后面有归一化的EX计算方法.
# EX 明确体现了其作用, EX可以用于:
    * 发现期权价格;
    * 用于预测VIX可能下降或者回升;
# 使用VIX计算SPY upper, 可以用来sell covered call

===========================================================
price data 从yahoo得到，最早到1993年
    * Price: close price after dividends and split
    * SPY returns, VIX returns: changes to yesterday
    * fwd returns：the thrid day open / the second day open - 1
    * fwd vol: next 21 days realized volatility
    * hist vol: past 21 days volatilty

我们可以看到相比hist vol，implied vol是对fwd vol更加准确的估计。

'''

# %%
figsize = (10, 8)
df = pd.read_csv('data' + os.sep + 'VIX_Study.csv',
                 parse_dates=True,
                 index_col=0).sort_index(ascending=True).round(2)
df['SPY returns'] = df['SPY Close'].pct_change()           # 当天收盘价变化
df['VIX returns'] = df['VIX'].pct_change()                 # 当天收盘价变化
df['fwd returns'] = df['SPY Open'].pct_change().shift(-2)  # 第三天开盘价 / 第二天开盘价 - 1
df['hist vol'] = df['SPY returns'].rolling(
    21).std() * np.sqrt(252) * 100  # 前21天年化波动率
df['fwd vol'] = df['SPY returns'].rolling(
    21).std().shift(-21) * np.sqrt(252) * 100  # 后21天年化波动率

volatilities = df[['VIX', 'hist vol', 'fwd vol']].dropna()
volatilities.corr()

# %%
sns.pairplot(volatilities)

# %%
df[['SPY returns', 'VIX returns']].corr()

# %%
print '''
先看一看VIX和fwd vol的价差。
* 如果这个价差为正的，表示投资者期待的vol大于实际的vol。投资者愿意出更多的钱买保险。
* 如果这个价差为负的，表示市场的风险比预期的更大。
'''
# %%
vix_fwd = volatilities['VIX'] - \
    volatilities['fwd vol']  # implied vol - real vol
vix_fwd.hist(bins=100)
plt.title('Implied Volatility - Realized Volatility')
plt.xlabel('% Difference')
plt.ylabel('Occurences')
plt.show()

# %%
print '''
价差是大部分正的，说明大部分时间期权以premium交易。
但是在左侧有很长的肥尾，说明有时候投资者大大低估了市场的波动率，比如黑天鹅事件。

下面我们绘出fwd vol被低估的极端时刻，

如果高亮这些波动率被低估的时期，我们发现波动率被低估的时期绝大部分在于Dot Com泡沫破裂
和全球金融危机期间。

可以看到这些低估主要发生在大的危机的起始阶段, 
表示人们还没有意识到危机的严重性.
'''
# %%
plt.figure(figsize=(10, 8))
# 最小2.5% implied vol - real vol
bottom_percentile = np.percentile(vix_fwd, 2.5)
worst_days = vix_fwd < bottom_percentile
df.dropna()['SPY Close'].plot()
df.dropna().loc[worst_days, 'SPY Close'].plot(style='ro')  # 注意这种过滤绘图方法
plt.show()

# %%
print '''
# Excess VIX
VIX - hist vol 我们称之为：Excess VIX, 简称EV。
归一化的EX我们在后面会看到计算方法, 先看一看未归一化的EX.

由于fwd vol在实际中是得不到的，现在我们看一看hist vol 和 implied vol的关系。
这两个指标在实际操作中可以得到， 所以特别重要。

正值表示投资者认为未来的波动率大于过去的波动率, 反之亦然.
和前面一样, implied vol通常是大于historical vol的. 

'''
# %%
vix_hist = volatilities['VIX'] - \
    volatilities['hist vol']  # implied vol - hist vol
vix_hist.hist(bins=100)
# %%
vix_hist.describe()
# %%
print '让我们看一看极端情况：'
print np.percentile(vix_hist, 99)  # =  13.20
print np.percentile(vix_hist, 1)  # = -10.75
# %%
print '''
但是讨厌的肥尾又出现了!
可以看到这些极端情况发生在市场的底部, 

# 表示在底部人们倾向于高估未来的波动率.
# 反过来说，如果EV足够大，说明大盘到达底部了 !
# 此时，其他做空波动率策略就可以开始启动了。
'''
# %%
plt.figure(figsize=figsize)
vix_hist_low = np.percentile(vix_hist, 1)   # implied vol 低于 hist vol 的极端情况
vix_hist_high = np.percentile(vix_hist, 99)  # implied vol 高于 hist vol 的极端情况
vix_hist_low_days = vix_hist <= vix_hist_low
vix_hist_high_days = vix_hist > vix_hist_high
df.dropna()['SPY Close'].plot()
df.dropna().loc[vix_hist_low_days,  'SPY Close'].plot(style='ro')  # 下跌加速
df.dropna().loc[vix_hist_high_days, 'SPY Close'].plot(style='go')  # 市场底部
plt.show()

# %%
print '''
我们再来看一看VIX对于SPY波动率的预测成功度.
VIX表示未来的年化波动率, 如果基于自然分布(这是不对的):
* price在68%的情况落在正负一个标准差之内.
* price在95%的情况落在正负两个个标准差之内.

我们看一看在2008年和2017年这两个极端年份的情况.

# 未来价格边界边界的计算可以预测21天后的SPY价格上下边界。 
# 对于 upper 来说，1个标准差已经非常安全的Sell coverd Call.
# 历史上如果SPY的价格高于1个标准差预测的价格，其实表明回调在即！

'''
# %%

std_num = 1
df['proj upper'] = df['SPY Close'].shift(  # Get upper SPY price based on VIX
    21) * (1 + std_num * df['VIX'].shift(21) / 100 * np.sqrt(21) / np.sqrt(252))
df['proj lower'] = df['SPY Close'].shift(  # Get lower SPY price based on VIX
    21) * (1 - std_num * df['VIX'].shift(21) / 100 * np.sqrt(21) / np.sqrt(252))
df.loc['2008', ['SPY Close', 'proj upper', 'proj lower']].plot(
    style=['b-', 'g:', 'r:'], figsize=figsize)
# %%
df.loc['2017', ['SPY Close', 'proj upper', 'proj lower']].plot(
    style=['b-', 'g:', 'r:'], figsize=figsize)

# %%
print '''
# VS Expected

下面我们看一看过去n天实际变化幅度和预测变化幅度的比较，我们称之为 vs expected。

# vs expected 可以在当天计算出，具有参考意义。
# n=5，vs expected = 一周以来大盘实际变化幅度 / 一周前预测的变化幅度
'''
# %%
n = 5
# SPY 与 n 天前相比实际价格变化幅度
expected_num = df['SPY Close'] - df['SPY Close'].shift(n)
# SPY 基于n天前VIX预测的价格变化幅度
expected_demon = df['SPY Close'].shift(
    n) * df['VIX'].shift(n) / 100 * np.sqrt(n) / np.sqrt(252)
df['vs expected'] = expected_num / expected_demon  # VS 实际价格变化幅度 / 预测价格变化幅度
df['vs expected'].hist(bins=100, figsize=figsize)

# %%
print df.loc[df['vs expected'] > 0, 'vs expected'].mean()  # 大盘上涨时
print df.loc[df['vs expected'] <= 0, 'vs expected'].mean()  # 大盘下跌时
print df.loc[df['vs expected'] > 0, 'vs expected'].count() / \
    float(df['vs expected'].count())

# %%

print '''
由一下QQ图可以看出, vs expected 分布具有肥尾效应，而且在大盘下跌时格外突出。
这说明人们在大跌的底部往往过度估计了价格下跌。
'''
# %%
import scipy.stats as stats
import pylab
stats.probplot(df['vs expected'].dropna(), dist='norm', plot=pylab)

# %% 2017年的 vs expected
df.loc['2017', 'vs expected'].plot(figsize=figsize)

# %%

print '''
仔细研究一下 vs expected 和大盘回报之间的规律：
将 vs expected 分成10个等分位，计算每个等分位的实际回报。
可以看到 在vs expected 值负得比较多时回报比较好。
如果在 vs expected 小于阈值 -0.5 的第二天开盘买入第三天开盘卖出，结果会怎么样？
* 阈值 -0.5 是可以调节的，越小回报的波动越小，但是交易机会越少，从而收益变少。
* 阈值 -0.5， sharpe 为1.7，回撤0.15， 是一个很不错的策略。
* 阈值 -0.4 也是一个不错的选项。

'''
# %%
df['fwd returns'].groupby(pd.qcut(df['vs expected'], 10)).mean().plot(
    kind='bar', figsize=figsize)

# %%
plt.figure(figsize=figsize)
eq = (1 + df.loc[df['vs expected'] < -0.5, 'fwd returns']).cumprod()  # 策略回报
eq.plot()  # equality curve
(1 + df['fwd returns']).cumprod().plot()  # 实际大盘
plt.show()
