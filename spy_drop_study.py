
# coding=utf-8
'''
在波动性急剧增大期间, 盘前下跌和盘中下跌的比较

通过2018年二月一号到四月七号之间的变化, 这2个多月的大盘下跌了9%. 
通过研究发现, 汇总来看,几乎所有下跌都是盘中实现的, 而盘前的变化基本能够互相抵消;

2008年的熊市这个现象很明显,达到了1:3. 
从2010年到现在, 盘前涨的比盘后多, 盘中跌的比盘前多, 这说明在牛市中盘前涨的也比盘中多.

这里表现出一个惊人的事实:从2000年到现在18年间, 盘中实际上仅仅涨了10%, 而另外90%都是盘前涨的.

结论

有3类股票:
1, 盘前上涨多,盘中下跌多:
   SPY
   QQQ, IWM, EEM, VNM, VHT, ITA, RHS的这个趋势更加强烈。
   AAPL, FB, GOOG, ZIV等股票的走势和SPY,QQQ一样,说明这些股票是大盘的驱动力.

2, 盘前下跌多, 盘中上涨多:
   比如: TLT, IEF, LMT, MO, STZ, 这说明这类股票和债券一样, 具有'避险'的性质.

3, 不明显的:
   DIA, IHI多半是少数股票的集合, 由于各种股票掺杂而且数量不多,不能形成明显趋势.
   MA, V, CME, 这些股票一般具有既有避险的性质,自己业绩也不错.

收获:
1, 大盘股, 大盘ETF的仓位不能离开市场, 要过夜, 特别是要大跌的尾盘杀入，大盘的均值回归在短期内非常强烈.
2, SDS等做对冲尽量不要过夜. SDS的对冲, 应该开盘进入, 尾盘卖出.

'''
# %%

import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
plt.style.use(['seaborn'])
# %%
# download stock price data into DataFrame
spy = 'SPY'

stock = data.DataReader(spy, 'yahoo', start='1/1/2000')
stock['last_close'] = stock['Close'].shift(1)
stock['change_before_market'] = (stock['Open'] - stock['last_close'])
stock['change_in_market'] = (stock['Close'] - stock['Open'])
stock = stock.dropna()

print stock.change_before_market.sum()
print stock.change_in_market.sum()

# %%
'''
这里表现出一个惊人的事实:从2000年到现在, 盘中实际上仅仅涨了10%, 而另外90%都是盘前涨的.
这说明两个道理:
1, long的仓位一定不能离开市场, 要过夜;
2, SDS仓位不能过夜,开盘进,尾盘出;
'''
plt.figure(figsize=(8, 5))
stock['cumsum_before_market'] = stock['change_before_market'].cumsum()
stock['cumsum_in_market'] = stock['change_in_market'].cumsum()
plt.plot(stock['cumsum_before_market'])
plt.plot(stock['cumsum_in_market'])
plt.hlines(y=0.0, xmin='1999', xmax='2019')
plt.legend()
plt.show()

# %%
plt.plot(stock.Close)
plt.show()


#%%
