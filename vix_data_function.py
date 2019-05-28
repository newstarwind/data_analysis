# %%
'''
处理目录的方法
'''
# -*- coding: utf-8 -*-
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'data'))
    print(os.getcwd())
except:
	print 'error'
# %%
'''
准备工作：解析html文档
Beautifulsoup 用于解析HTML文档
文档： https://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/
'''
import urllib2
from bs4 import BeautifulSoup
def getVixFutures():
    '''
    return (date, f1 vix, f2 vix)
    '''
    page = urllib2.urlopen('http://vixcentral.com/historical/?days=1')
    soup = BeautifulSoup(page)
    datadate = '2000-01-01'
    f1=f2= 0.0
    for i, incident in enumerate(soup('tr')[1]):
        if i == 0:
            datadate = incident.text
        elif i == 2:
            f1 = float(incident.text)
        elif i == 3:
            f2 = float(incident.text)
    return (datadate, f1, f2)

print getVixFutures()    
#%%
def getCurrentVix():
    '''
    return (date, current, high, low)
    '''
    page = urllib2.urlopen('http://www.stockq.cn/index/VIX.php')
    soup = BeautifulSoup(page)
    tr   = soup.find_all('tr', class_='row2', limit=1)[0]
    result =[]
    for td in tr.children:
        if td.name: 
            result.append(td.text)
    return (result[7],result[0],result[3],result[4])
print getCurrentVix()
#%%
