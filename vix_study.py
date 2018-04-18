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
===========================================================

从本研究我们可以发现:
# EX : VIX - hist vol：Excess VIX, 简称EV。后面有归一化的EX计算方法.
    * 发现期权价格;
    * 用于预测VIX可能下降或者回升;

# 使用VIX计算SPY upper, 可以用来sell covered call

# vs expected 可以在当天计算出，具有参考意义。
    * n=5，vs expected = 一周以来大盘实际变化幅度 / 一周前预测的变化幅度
    * vs expected < -0.5 具有指标意义，表示良好的进入时机。

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
vix_fwd.describe()

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

# %%


def sharpe_ratio(returns):
    return returns.mean() / returns.std()


def drawdown(eq):
    return (eq / eq.cummax() - 1)


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
    win_pct = num_gains / num_total
    avg_win = np.mean(gains)
    avg_loss = np.mean(losses)
    stats['total returns'] = num_total
    stats['total gains'] = num_gains
    stats['total losses'] = num_losses
    stats['expectancy (%)'] = avg * 100
    stats['volatilty (%)'] = volatility * 100
    stats['sharpe (daily)'] = sharpe
    stats['win %'] = win_pct * 100
    stats['total returns'] = num_total
    stats['average gain (%)'] = avg_win * 100
    stats['average loss (%)'] = avg_loss * 100
    return stats


print(sharpe_ratio(df['fwd returns']) * np.sqrt(252))  # 大盘年化sharpe
print(sharpe_ratio(df.loc[df['vs expected'] < -0.5,
                          'fwd returns']) * np.sqrt(252))  # 策略年化sharpe

# %% Draw the drawdown diagram
plt.figure(figsize=figsize)
drawdown(eq).plot()  # 策略DD
df['SPY DD'] = drawdown(df['SPY Close'])  # 大盘DD
df['SPY DD'].plot()

# %%
over_40 = df.loc[df['VIX'] > 40]
df['2008']['SPY Close'].plot(figsize=figsize)  # 2008年大盘
df.loc[over_40.index, 'SPY Close']['2008'].plot(
    style='ro', figsize=figsize)  # VIX>40红点
df['SPY Close'].rolling(200).mean()['2008'].plot(figsize=figsize)  # 200 MA

# %%
over_40.loc[over_40['fwd returns'] <= 0, 'fwd returns'].describe()
# %%
over_vix_under_ma = (df['VIX'] > 35) & (  # vix > 35 and price > 200 MA
    df['SPY Close'] > df['SPY Close'].rolling(200).mean())
df.loc[over_vix_under_ma, 'fwd returns']
# %% 如果在 VIX  < hist vol 时第二条开盘入,第三天开盘出, 结果如何?
strat = df.loc[df['VIX'] < df['hist vol'], 'fwd returns']
pd.DataFrame({'strat': summary_stats(strat),
              'SPY': summary_stats(df['fwd returns'])})
# %% 求最优化解, 求hist vol和VIX的: alpha = 7.69, beta = 0.74
'''
使用hist vol 推导 VIX
'''
import statsmodels.api as sm
X = df.dropna()['hist vol']
X = sm.add_constant(X)
y = df.dropna()['VIX']
model = sm.OLS(y, X).fit()
model.params

# %% 表示近74%的 implied vol能够用hist vol解释
model.summary()
# %% 图解hist vol和VIX的关系, 绘制线性拟合
plt.figure(figsize=figsize)
# 线性模型的根据hist vol 对于VIX的预测值
historical_component = df['hist vol'] * model.params[1] + model.params[0]
plt.scatter(df['hist vol'], df['VIX'])
plt.plot(df['hist vol'], historical_component, color='r')
plt.xlabel('hist vol')
plt.ylabel('VIX')


# %% 进行归一化处理
print '''
EX 归一化

VIX中最有意思的是不能使用hist vol解释的部分.
为了得到这一部分信息, 我们将观察模型的残差 residuals.
我们计算 实际值 - 线性模型预测的值 , 并将其归一化到[-1,1] 之间, 得到残差. 
EX 越大表示 VIX值中不能被hist vol解释的成分越多.
'''
# %% plot residuals 残差
resid = df['VIX'] - historical_component
resid.plot(figsize=figsize)
plt.axhline(resid.mean(), color='g')
plt.title('VIX Residuals')


# %%
def obj_func(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# 归一化方法


def normalize(x):
    return (x - np.mean(x)) / np.std(x)

# 高低值差距设为1


def bound(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


normalized = normalize(resid)
bounded = bound(normalized)   # 高低值差距为1
bounded = bounded * 2 - 1     # 展开到以0为横轴的[-1, 1]区间
bounded.plot(figsize=figsize)  # 绘制归一化的EX
plt.title('Scaled Residuals: EX')

# %%
print '''
当VIX 比 根据 vist vol 得到估算值高很多时, 是一个很好的大盘买入时机.
我们考察一下EX高的时机买入的收益情况.
'''
# %%  根据EX 10 分位对fwd returns汇总
df['fwd returns'].groupby(pd.qcut(bounded, 10)).mean().plot(
    kind='bar', figsize=figsize)
plt.title('Mean Return by EX Decile')
plt.xlabel('Scaled EX')
plt.ylabel('Mean Daily SPY Return')
# %%
compare_df = pd.DataFrame()
compare_df['SPY'] = summary_stats(df['fwd returns'])
compare_df['Top Decile'] = summary_stats(
    df.loc[bounded > bounded.quantile(0.9), 'fwd returns'])
compare_df

# %%
plt.figure(figsize=figsize)
# 取EX的最大90分位值
# 为了计算扩张窗口平均（expanding window mean），我们要使用expanding操作符，而不是用rolling。
# 这个扩张平均的时间窗口是从时间序列开始的地方作为开始，窗口的大小会逐渐递增，直到包含整个序列。
expanding_quantile = bounded.expanding(min_periods=10).quantile(0.9)
top_quantile = bounded > expanding_quantile
# 按照EX大于90分位值过滤
filtered = df.loc[top_quantile, 'fwd returns']
# 按照VIX大于30过滤
filtered_2 = df.loc[df['VIX'] > 30, 'fwd returns']
# 按照EX > 0 过滤
filtered_3 = df.loc[bounded > 0, 'fwd returns']

results = pd.DataFrame()
results['SPY'] = summary_stats(df['fwd returns'])
results['Top Quantile'] = summary_stats(filtered)
results['Excess > 0'] = summary_stats(filtered_3)
results['High VIX'] = summary_stats(filtered_2)
(1 + filtered).cumprod().plot()
(1 + filtered_2).cumprod().plot()
(1 + filtered_3).cumprod().plot()
(1 + df['fwd returns']).cumprod().plot()
plt.legend(['Top Quantile', 'High VIX', 'EX > 0', 'SPY'])

# %%
results
# %%
num_days = (bounded.index[-1] - bounded.index[0]).days
num_years = num_days / 365.25
total_return = (1 + 2 * filtered).cumprod().dropna()
cagr = (total_return.iloc[-1] / total_return.iloc[0]) ** (1/num_years) - 1
max_dd = drawdown(total_return).min()
print('CAGR:{}'.format(cagr))
print('Max DD: {}'.format(max_dd))
# %%
(df['SPY Close'].iloc[-1] / df['SPY Close'].iloc[0]) ** (1 / num_years) - 1

# %%
print '''
EX 明确体现了其作用, EX可以用于:
1, 发现期权价格;
2, 用于预测VIX可能下降或者回升;
'''
