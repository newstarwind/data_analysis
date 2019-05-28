'''
study relations between vix and spy
'''

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
plt.style.use(['seaborn'])
figsize = (10,8)

# %%
df = pd.read_csv('data' + os.sep + 'VIX_Study.csv',
                parse_dates=True,
                index_col=0).sort_index(ascending=True).round(2)

df.describe()
#%%
df['SPY return'] = df['SPY Close'].pct_change() # 当天收盘价变化
df['VIX return'] = df['VIX'].pct_change() # 当天收盘价变化
df['fwd return'] = (df['SPY Open'].shift(-2) - df['SPY Open'].shift(-1))/df['SPY Open'].shift(-1) # 第二天开盘买入第三天开盘卖出
df['fwd close buy return'] = (df['SPY Close'].shift(-1) - df['SPY Close']) / df['SPY Close'] #马上闭市买入，第二天收盘卖出
df['hist vol']   = df['SPY return'].rolling(21).std() * np.sqrt(252) * 100
df['fwd vol']    = df['SPY return'].shift(-21).rolling(21).std() * np.sqrt(252)*100

df = df.dropna()
vols = df[['VIX','hist vol','fwd vol']]
vols.corr() # We can see fwd vol has higher relation with VIX, not hist vol
df[['SPY return','VIX return']].corr() # You can see vix beta is about -8

#%%
sns.pairplot(vols)

#%%
'''
EX: 波动率溢价
* 如果EX为正的，表示投资者认为的vol大于实际的vol。投资者愿意出更多的钱买保险。
* 如果EX为负的，表示市场的风险比预期的更大。
'''
ex = df['VIX']-df['fwd vol']
ex.hist(bins=100)
plt.title('implied vol - realized vol')
plt.xlabel('%Difference')
plt.ylabel('Occurance')

#%%
'''
EX是大部分正的，说明大部分时间期权以premium交易。
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
归一化的EV我们在后面会看到计算方法, 先看一看未归一化的EX.

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

df['vs_expected'] = real_change/expect_change  # 实际价格变化幅度 / 预测价格变化幅度
df['vs_expected'].hist(bins = 100)

print df.loc[df['vs_expected']>0, 'vs_expected'].mean() # 大盘上涨
print df.loc[df['vs_expected']<0, 'vs_expected'].mean() # 大盘下跌

upper_correct = df['vs_expected'] <  1
down_correct  = df['vs_expected'] > -1
expected_ok   = upper_correct & down_correct
expected_ok_percent = df.loc[expected_ok,'vs_expected'].count() / float(df['vs_expected'].count()) * 100
print 'correct percent is %s' % expected_ok_percent
#%%
'''
由一下QQ图可以看出, vs expected 分布具有肥尾效应，而且在大盘下跌时格外突出。
这说明人们在大跌的发展过程（提前n天）中估计价格下跌不足。
'''
import scipy.stats as stats
import pylab
stats.probplot(df['vs_expected'].dropna(), dist='norm', plot=pylab)

#%%
'''
仔细研究一下 vs expected 和大盘回报之间的规律：
将 vs expected 分成10个等分位，计算每个等分位的实际回报。
可以看到 在vs expected 值负得比较多时回报比较好。
如果在 vs expected 小于阈值 -0.4 的第二天开盘买入第三天开盘卖出，结果会怎么样？
* 阈值 -0.4 是可以调节的，越小回报的波动越小，但是交易机会越少，从而收益变少。
* 阈值 -0.4， sharpe 为1.7，回撤0.15， 是一个很不错的策略。
* 阈值 -0.4 也是一个不错的选项。

pd.qcut(df['vs_expected'],10) #基于百分位的离散化功能, 根据样本值将之纳入不同百分位范围。返回Series
* 基于vs_expected百分位进行回报统计平均回报, 假设第二天开盘买入第三天开盘卖出
* 可见最低百分位的回报率比较高，明显高于总平均值

'''
df['fwd return'].groupby(pd.qcut(df['vs_expected'], 10)).mean().plot( 
    kind='bar')
eq = (1 + df.loc[df['vs_expected'] < -0.4, 'fwd return']).cumprod()  # 策略回报
eq.plot()  # equality curve
(1 + df['fwd return']).cumprod().plot()  # 实际大盘
plt.show()

#%%
'''
如果在 vs expected 小于阈值 -0.4 的当天闭市价买入，也是第二天开盘卖出，结果会怎么样？
这和我的另外一项研究吻合：SPY的上涨大部分时间是在盘前完成的。
'''
df['fwd close buy return'].groupby(pd.qcut(df['vs_expected'], 10)).mean().plot( 
    kind='bar')
eq = (1 + df.loc[df['vs_expected'] < -0.4, 'fwd close buy return']).cumprod()  # 策略回报
eq.plot()  # equality curve
(1 + df['fwd return']).cumprod().plot()  # 实际大盘
plt.show()


#%%
'''
下面我们仔细看一看这个策略的指标

'''
# Calculating the Sharpe ratio using daily returns is easier than computing the monthly ratio. 
# The average of the daily returns is divided by the sampled standard deviation of the daily returns 
# and that result is multiplied by the square root of 252–the typical number of trading days per year 
def sharpe_ratio(returns):
    return returns.mean()/returns.std()

def summary_stats(returns):
    stats = pd.Series()
    gains = returns[returns > 0]
    losses = returns[returns <= 0]
    num_total = len(returns)
    num_gains = len(gains)
    num_losses = len(losses)
    avg = np.mean(returns)
    volatility = np.std(returns)
    sharpe = avg / volatility
    win_pct = num_gains / float(num_total)
    avg_win = np.mean(gains)
    avg_loss = np.mean(losses)
    stats['total trades']      = num_total
    stats['total gain times']  = num_gains
    stats['total loss times']  = num_losses
    stats['win % ']            = round(win_pct * 100,2)
    stats['average gain (%)']  = avg_win * 100
    stats['average loss (%)']  = avg_loss * 100
    stats['expectency (%)']    = avg * 100
    stats['volatility (%)']    = volatility * 100
    stats['sharpe']            = round(sharpe * np.sqrt(252),2)
    stats['total returns (%)'] = (1+returns).cumprod()[-1] * 100
    return stats

print '大盘'
print summary_stats(df['fwd return'])
print '\n\n第二天开盘买入，第三天开盘卖出'
print summary_stats(df.loc[df['vs_expected']< -0.5, 'fwd return'])
print '\n\n当天马上闭市价入，第三天开盘出， sharpe ratio'
print summary_stats(df.loc[df['vs_expected']< -0.5, 'fwd close buy return'])

# %%
df.loc['2017', 'vs_expected'].plot()

#%%
'''
我们看一看回撤的情况
当天收盘买入的回撤较大，在2002年和2009年达到了25%
第二天开盘买入的回撤相对小一些，在20%左右
当然它们都比大盘小很多，而且时间上也不同步
'''
# 计算当前值和之前最大值的pct_change
def drawdown(eq):
    return eq / eq.cummax() - 1

eq1 = (1 + df.loc[df['vs_expected'] < -0.4, 'fwd return']).cumprod()  # 第二天开盘买入， 策略回报
eq2 = (1 + df.loc[df['vs_expected'] < -0.4, 'fwd close buy return']).cumprod() # 当天收盘买入，策略回报
drawdown(eq1).plot(style='r-', figsize=(20,16))
drawdown(eq2).plot(style='g-',figsize=(20,16))
df['SPY DD'] = drawdown(df['SPY Close'])    
df['SPY DD'].plot()


#%%
'''
我们看一看如果在VIX>40时购入SPY，会怎么样？
'''
over_40 = df.loc[df['VIX'] > 40]
df.loc['2008']['SPY Close'].plot()
df.loc[over_40.index, 'SPY Close'].loc['2008'].plot(style='ro')
df['SPY Close'].rolling(200).mean().loc['2008'].plot()

#%%
'''
# 如果VIX>40时买入了，亏钱的情况如何？
一般亏2%，最多一次亏12%，还是有一点吓人的.
'''
over_40.loc[over_40['fwd return'] <= 0, 'fwd return'].describe()


#%%
'''
# 如果在VIX > 35, 并且SPY收盘价在200天线以上时买入，会怎么样？
这样的情况并不多，大概只有10天，因为大多数时候SPY在200天线以上的时候，VIX不会上到35
'''
over_vix_under_ma = (df['VIX'] > 35) & (df['SPY Close'] > df['SPY Close'].rolling(200).mean())
df.loc[over_vix_under_ma, 'fwd return'].count()

#%%
'''
# 如果一个策略是在VIX < hist vol 时买入大盘
我们将这个策略称为strat，与大盘比较一下。
看来这个策略并不好，远远差于大盘。
'''
strat = df.loc[df['VIX'] < df['hist vol'], 'fwd return']
pd.DataFrame({'strat':summary_stats(strat), 'SPY':summary_stats(df['fwd return'])})



#%%
'''
# 让我们使用线性回归,探讨hist vol和VIX的关系, 发现：
VIX = 7.71 + hist vol * 0.745
'''
import statsmodels.api as sm
X = df.dropna()['hist vol']
X = sm.add_constant(X)
y = df.dropna()['VIX']
model = sm.OLS(y, X).fit()
print model.params[0]
print model.params[1]

historical_component = df['hist vol'] * model.params[1] + model.params[0]
plt.scatter(df['hist vol'], df['VIX']) # 绘制VIX和hist vol对应的散点图
plt.plot(df['hist vol'], historical_component, color='r') # 根据用一个自变量， 绘制线性回归线


#%%
'''
看一看VIX和理论模型之间的差值
'''
# 计算差值
resid = df['VIX'] - historical_component
resid.plot()
plt.axhline(resid.mean(), color='b')
plt.title('VIX Residuals')

#%%
'''
下面我们将VIX 差值进行归一化处理
'''
# 数据归一化
def normalize(x):
    return (x - np.mean(x)) / np.std(x)
# 将归一化之后的数据值归纳到[0,1]之间
def bound(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

normalized = normalize(resid)
bounded = bound(normalized)
bounded = bounded * 2 - 1 #将数据归纳到[-1,1]之间
bounded.plot()
plt.title('Scaled Residuals')


#%%
'''
这个百分位研究显示，在Excess VIX处于高位，也就是人们对于未来波动性的估计大大高于历史波动性的时候
在第二天买入并且在第三天开盘抛出，收益最高。
'''
df['fwd return'].groupby(pd.qcut(bounded, 10)).mean().plot(kind='bar')
plt.title('Mean Return by Excess VIX Decile')
plt.xlabel('Scaled Excess VIX')
plt.ylabel('Mean Daily SPY Return')


#%%
'''
Top Decile 策略的收益不如大盘，但是sharpe ratio为1.68，远高于大盘，可以作为短线策略
'''
compare_df = pd.DataFrame()
compare_df['SPY'] = summary_stats(df['fwd return']) # 大盘
compare_df['Top Decile'] = summary_stats(df.loc[bounded > bounded.quantile(0.9), 'fwd return']) # Excess VIX 最大十分位
compare_df

#%%
'''
看一看策略的收益曲线，可以看到由于百分位为0.9,大部分时间在空仓，收益并不如意。
'''
eq = (1 + df.loc[bounded > bounded.quantile(0.9), 'fwd return']).cumprod()  # 策略回报
eq.plot()  # equality curve
(1 + df['fwd return']).cumprod().plot()  # 实际大盘
plt.show()

#%%
'''
让我们将本篇见到的各种策略进行一个比较
可以看到，最好的还是vs expected策略, 闭市买入闭市出

本质而言， vs expected 和 Top Decile 是对于一个想法的不同表述
但是， vs expected的表述远为简单并且可以进一步定制

Excess VIX > 0 的收益和持有大盘类似，sharpe 也不高
Top Decile 条件严格，所以收益并不好

'''
expanding_quantile = bounded.expanding(min_periods=10).quantile(0.9)
top_quantile = bounded > expanding_quantile
filtered = df.loc[top_quantile, 'fwd return']    # Top Decile
filtered_2 = df.loc[df['VIX']>30, 'fwd return']  # Top VIX
filtered_3 = df.loc[bounded > 0, 'fwd return']   # Excess  VIX > 0
filtered_4 = df.loc[df['vs_expected'] < -0.4, 'fwd close buy return'] # vs expected n = 5
results = pd.DataFrame()
results['SPY']          = summary_stats(df['fwd return'])
results['Top Decile']   = summary_stats(filtered)
results['Excess > 0']   = summary_stats(filtered_3)
results['High VIX']     = summary_stats(filtered_2)
results['vs expected']  = summary_stats(filtered_4)
plt.figure(figsize=(20,16))
(1 + filtered).cumprod().plot() # Top Decile
(1 + filtered_2).cumprod().plot()  # Top VIX
(1 + filtered_3).cumprod().plot(style = 'g') # Excess VIX > 0 
(1 + filtered_4).cumprod().plot(style = 'r') # vs expected
(1 + df['fwd return']).cumprod().plot(style = 'b') # 大盘
results

#%%
'''
看一看vs expected策略的年收益
1993 - 2018/2, CAGR = 8.9%
MDD: 26.7%
基本可以说穿越牛熊
'''
num_days = (filtered_4.index[-1] - filtered_4.index[0]).days
num_years = num_days / 365.25
total_return = (1 + filtered_4).cumprod().dropna()
cagr = (total_return.iloc[-1] / total_return.iloc[0]) ** (1 / num_years) - 1
max_dd = drawdown(total_return).min()
spy_cagr = (df['SPY Close'].iloc[-1] / df['SPY Close'].iloc[0]) ** (1 / num_years) - 1
spy_mdd = drawdown(df['SPY Close']).min()

print('SPY CAGR: {}'.format(spy_cagr))
print('vs expected CAGR: {}'.format(cagr))
print('Max DD: {}'.format(max_dd))
print('SPY MDD: {}'.format(spy_mdd))

#%%
