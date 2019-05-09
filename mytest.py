
#%%

from pandas import DataFrame, Series
import pandas as pd
import numpy as np

df = DataFrame({'key1':['a','a','b','b','a'],
                'key2':['one','two','one','two','one'],
                'data1':np.random.randn(5),
                'data2':np.random.randn(5)})

print df
# print df['key1']
# print df[:2]['key1']
print df.loc[2,'key1']
print df.iloc[2,2]
#%%
foo = Series({'k1':1,'k2':2})
bar = Series({'k1':1,'k3':3})

foo.add(bar,fill_value=0)

#%%
f = lambda x: x.max() - x.min()
df = DataFrame({'k1':np.random.randn(5),
                'k2':np.random.randn(5)})

df.apply(f,axis = 1)



#%%
f = lambda x,y=10:x+y
f(3)


#%%
