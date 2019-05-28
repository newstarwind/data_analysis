# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %%
df = pd.read_csv('data' + os.sep + 'spy.csv',
                 parse_dates=True,
                 index_col=0).sort_index(ascending=True).round(2)


df = df.resample('D', how = 'last').dropna()
df['log_ret'] = np.log(df).diff()
log_ret = df['log_ret']
spy = log_ret.dropna()
spy.head()


# %%
print '''
"Sell in May and go away" 
让我们看一看夏天的股市表现和其他月份的股市表现差异
'''

# %%
df = spy.to_frame(name='pct_chg') # convert Series to DataFrame
by_month = df.resample('BM').sum() # sum of log_return by month
by_month['month'] = by_month.index.month


title='Avg Log Return (%): by Calendar Month (2006-present)'
ax = (by_month.groupby('month').pct_chg.mean()*100).plot\
.bar(color='grey',title=title)
ax.axhline(y=0.00, color='grey', linestyle='--', lw=2) # horizan line

# %%
by_month['season'] = None
by_month.loc[ by_month.month.between(5,10),'season'] = 'may_oct'
by_month.loc[~by_month.month.between(5,10),'season'] = 'nov_apr'

(by_month.groupby('season').pct_chg.mean()*100).plot.bar\
(title='Avg Monthly Log Return (%): \nMay-Oct vs Nov_Apr (2006-present)'
 ,color='grey')

# %%
print '''
夏季和非夏季月份的股市表现有显著差别.
'''

# %%
title='Avg Monthly Log Return (%) by Season\nFour Year Moving Average (2006-present)'
by_month['year'] = by_month.index.year
ax = (by_month.groupby(['year','season']).pct_chg.mean().unstack().rolling(4).mean()*100).plot(title=title)
ax.axhline(y=0.00, color='grey', linestyle='--', lw=2)

# %%
print '''
有个理论认为这个现象和选举年有关系, 下面按照大选年来group夏天和非夏天的用户.
通过以下的分析可以看出, 在大选年, 夏天股市不振的情况更加显著. 

'''

# %%
election_years = ['2004','2008','2012','2016']
by_month['election_year'] = False
by_month.loc[by_month.year.isin(election_years),'election_year'] = True

title = 'Average Monthly Returns (log) for Non-Election Years vs. Election Years'
ax = (by_month.groupby(['election_year','season']).pct_chg.mean()\
      .unstack()*100).plot.bar(title=title)
ax.axhline(y=0.00, color='grey', linestyle='--', lw=2)

# %%
print '''
下面我们研究在一周中股价涨跌分布的模式
可见周2,周3最好, 所以加仓要在周1或者周4周5
'''
# %%
df = spy.to_frame(name='pct_chg') # Series to DataFrame
by_day = df 
by_day['day_of_week'] = by_day.index.weekday + 1 # add week column

(by_day.groupby('day_of_week').pct_chg.mean()*100).plot.bar\
(title='Avg Daily Log Return (%): by Day of Week (2006-present)',color='grey')

# %%
# 观察周一到周三, 以及周四周五这两组和情况
by_day['part_of_week'] = None
by_day.loc[by_day.day_of_week <=3,'part_of_week'] = 'mon_weds'
by_day.loc[by_day.day_of_week >3,'part_of_week'] = 'thu_fri'

(by_day.groupby('part_of_week').pct_chg.mean()*100).plot.bar\
(title='Avg Daily Log Return (%): \nMon-Wed vs Thu-Fri (1993-present)'\
 ,color='grey')
# %%
# 观察滚动4年期间的模式
title='Avg Daily Log Return (%) by Part of Week\nFour Year Moving Average (2006-present)'
by_day['year'] = by_day.index.year
ax = (by_day.groupby(['year','part_of_week']).pct_chg.mean().unstack().rolling(4).mean()*100).plot()
ax.axhline(y=0.00, color='grey', linestyle='--', lw=2)
ax.set_title(title)

# %%
print '''
现在我们看一看在一个月中价格涨跌的分布,
看起来 "good days" (28-5, 11-18) and the "other days"
'''
# %%
by_day['day_of_month'] = by_day.index.day 

title='Avg Daily Log Return (%): by Day of Month (2006-present)'

ax = (by_day.groupby('day_of_month').pct_chg.mean()*100).plot(xlim=(1,31),title=title)
ax.axhline(y=0.00, color='grey', linestyle='--', lw=2)

# %%

by_day['part_of_month'] = None
good_days = [1,2,3,4,5,11,12,13,14,15,16,17,18,28,29,30,31]
by_day.loc[by_day.day_of_month.isin(good_days),'part_of_month'] = 'good_days'
by_day.loc[~by_day.day_of_month.isin(good_days),'part_of_month'] = 'other_days'

(by_day.groupby('part_of_month').pct_chg.mean()*100).plot.bar\
(title='Avg Daily Log Return (%): \nDays 1-5, 11-18, 28-31 vs Others (2006-present)'\
 ,color='grey')

# %%


title='Avg Daily Log Return (%) by Part of Month\nFour Year Moving Average (1993-present)'
by_day['year'] = by_day.index.year
ax = (by_day.groupby(['year','part_of_month']).pct_chg.mean().unstack().rolling(4).mean()*100).plot(title=title)
ax.axhline(y=0.00, color='grey', linestyle='--', lw=2)

# %%
print(by_day.groupby(['year','part_of_month']).pct_chg.mean().unstack().tail(5)*100)