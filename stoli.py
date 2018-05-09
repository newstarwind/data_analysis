# Stock Indicators Library 
def Bollinger(df, name, devs=2, w=20):
    """Returns bands as three new columns"""
    df[name + '_MID'] = df[name].rolling(w).mean()
    df[name + '_HI'] = df[name + '_MID'] + df[name].rolling(w).std() * devs
    df[name + '_LO'] = df[name + '_MID'] - df[name].rolling(w).std() * devs

def RSI(df, name, w=14):
    """Returns RSI as new column"""
    import pandas as pd
    temp = pd.DataFrame(index=df.index, columns=['Delta', 'Gain','Loss', 'MeanG', 'MeanL', 'RS'])
    temp['Delta'] = df[name] - df[name].shift(1, axis=0)
    temp['Gain'] = temp['Delta'][temp['Delta'] > 0] 
    temp['Loss'] = -1 * temp['Delta'][temp['Delta'] < 0]
    temp = temp.fillna(0)
    temp['MeanG'] = temp['Gain'].ewm(span=(2*w-1)).mean()
    temp['MeanL'] = temp['Loss'].ewm(span=(2*w-1)).mean()
    temp['RS'] = temp['MeanG'] / temp['MeanL'] 
    df[name + '_RSI'] = 100 - 100 / (1 + temp['RS'])
    
def MACD(df, name, short=12, long=29, signal=9):
    """Returns Three MACD Lines as new columns"""
    df[name + '_MACD'] = df[name].ewm(span=short).mean() - df[name].ewm(span=long).mean()
    df[name + '_Signal'] = df[name + '_MACD'].ewm(span=signal).mean()
    df[name + '_Histo'] = df[name + '_MACD'] - df[name + '_Signal'] 
    
def ATR(df, name, w=14):
    """df must include High and Low"""
    import pandas as pd
    temp = pd.DataFrame(index=df.index, columns=['A','B','C','TR'])
    temp['A'] = df['High'] - df['Low'] 
    temp['B'] = abs(df['High'] - df[name].shift(1, axis=0))
    temp['C'] = abs(df['Low'] - df[name].shift(1, axis=0))
    temp['TR'] = temp[['A', 'B', 'C']].max(axis=1)
    df[name + '_ATR'] = temp['TR'].ewm(span=(2*w-1)).mean()
    
def Keltner(df, name, atrs=1.5, a=10, w=20):
    """Returns top and bottom as two new columns"""
    ATR(df, name, a)
    df[name + '_HIK'] = df[name].ewm(span=w).mean() + atrs * df[name + '_ATR']
    df[name + '_LOK'] = df[name].ewm(span=w).mean() - atrs * df[name + '_ATR']
        
def ADX(df, name, w=14):
    """df must include High and Low"""
    import pandas as pd
    temp = pd.DataFrame(index=df.index, columns=['+DM','-DM'])
    ATR(df, name, w)
    temp['dHI'] = df['High'] - df['High'].shift(1, axis=0)
    temp['dLO'] = df['Low'] - df['Low'].shift(1, axis=0) 
    temp['POS'] = temp['dHI'] + temp['dLO']
    temp['+DM'] = temp['dHI'][temp['POS'] > 0] 
    temp['-DM'] = -temp['dLO'][temp['POS'] < 0] 
    temp['+DM'] = temp['+DM'][temp['+DM'] > 0]
    temp['-DM'] = temp['-DM'][temp['-DM'] > 0]
    temp.replace(to_replace='NaN', value=0, inplace=True)
    temp['+DM14'] = temp['+DM'].ewm(span=(2*w-1)).mean()
    temp['-DM14'] = temp['-DM'].ewm(span=(2*w-1)).mean()
    temp['+DI14'] = 100 * temp['+DM14'] / df[name + '_ATR'] 
    temp['-DI14'] = 100 * temp['-DM14'] / df[name + '_ATR'] 
    temp['Diff'] = abs(temp['+DI14'] - temp['-DI14'])
    temp['Sum'] = temp['+DI14'] + temp['-DI14']
    temp['DX'] = 100 * temp['Diff'] / temp['Sum'] 
    df[name + '_ADX'] = temp['DX'].ewm(span=(2*w-1)).mean()        
    
def STO(df, name, w=14, s=3):
    """Returns STO line and D as two new columns"""
    import pandas as pd
    temp = pd.DataFrame(index=df.index, columns=['HH', 'LL'])
    temp['HH'] = df['High'].rolling(w).max()
    temp['LL'] = df['Low'].rolling(w).min()
    temp['K'] = (df[name] - temp['LL'])/(temp['HH'] - temp['LL']) * 100
    df[name + '_STO'] = temp['K'].rolling(s).mean()
    df[name + '_D'] = df[name + '_STO'].rolling(s).mean()    