
#%%

from pandas import DataFrame, Series
import pandas as pd
import numpy as np

df = DataFrame({'key1':['a','a','b','b','a'],
                'key2':['one','two','one','two','one'],
                'data1':np.random.randn(5),
                'data2':np.random.randn(5)})
#%%
#DF的各种属性
print df.values
print df.columns
print df.shape
print df.index
print df.dtypes
print df.head(2)
print df.tail(2)
#%%
# 各种切片和定位方式
print df
print df['key1']
print df[:2]['key1']
# 推荐使用loc和iloc两个方法，注意需要先指定行，然后列（optional）两个参数
print df.loc[2,'key1']
print df.iloc[2,2]
#%%
foo = Series({'k1':1,'k2':2})
bar = Series({'k1':1,'k3':3})

# 两个DataFrame或者Series可以进行算术运算
foo.add(bar,fill_value=0)
#%%
f = lambda x: x.max() - x.min()
df = DataFrame({'k1':np.random.randn(5),
                'k2':np.random.randn(5)})
# apply方法可以用于对容器内的数据进行运算，
# 注意此时f函数的输入是Series，而不是单个数据
print df.apply(f,axis = 0)

#%%
f = lambda x,y=10:x+y
f(3)
#%%
df.sort_index(by='k1')
#%%
# DF常用的统计函数
# print df.mean(axis = 1)
# print df.median(axis = 1)
# print df.sum() #求和
# print df.describe()  #输出DF或者Series的概况信息
# print df.std(axis = 0) #标准差
# print df.cumsum(axis = 0) #计算累积和
# print df.cumprod(axis = 0) #计算累积积
# print df.diff(axPis=0)#当前行-上一行
print df.pct_change(axis =0)
#%%
df = DataFrame(data = np.arange(8).reshape(4,2), columns=['a','b'])
df2 = df.set_index('b') #设置b列为index
print df2
df2 = df.reset_index()  #将index列恢复成正常列
print df2

#%%
df = pd.DataFrame(np.random.randn(6,4),columns=list('ABCD'))
print df
print df[df.A > 1] # 以列的值进行过滤
print df[df.D < 0]

#%%
# 遍历所有行
for row in df.iterrows() :
    print row
# 遍历所有行,并包含index的值，一行作为一个tuple返回
for row in df.itertuples():
    print row

# 遍历所有列
for column in df.columns:
    print df[column]

#%%
