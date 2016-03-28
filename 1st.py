# -*- coding: utf-8 -*-
"""
Created on Sat Oct 04 16:15:28 2014

@author: Saurabh
"""
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from ggplot import *
from datetime import datetime
# <codecell>

print sm.datasets.sunspots.NOTE


dta = sm.datasets.sunspots.load_pandas().data

# <codecell>

dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]
dta.plot(figsize=(12,8));


## read a time series value in data
t1 =pd.read_csv("F:\\Data2\\LA\\Ass2\\No_password_Fin.csv" , index_col='Date' , parse_dates=True)

t2=t1[[4]]
T2=t1[[4]].values

t2.plot(figsize=(12,8));

movg = pd.rolling_mean(t2, 40)

t2.plot(label='Time_series')
movg.plot(label='movg')

#json1 = d4.to_json(orient="records")
#
#print json.dumps(json1, indent=4)
#f= open("F:\\Data3\\Unix\\output2.json","w+")
#f.write(json1)
#f.close()
#import sys
#sys.path.append('c:\\anaconda\\lib\\site-packages') 
#sys.path.append('C:\\Users\\Saurabh\\Anaconda\\Scripts')
#import numpy as np
#from matplotlib import pyplot as plt
#from nitime import utils
#from nitime import algorithms as alg
#from nitime.timeseries import TimeSeries
#from nitime.viz import plot_tseries
#npts = 2048
#sigma = 0.1
#drop_transients = 128
#Fs = 365
#coefs = np.array([0.9, -0.5])
#ts_x = TimeSeries(t2, sampling_rate=Fs, time_unit='s')
#fig01 = plot_tseries(ts_x, label='AR signal')
#TS = UniformTimeSeries(t2,sampling_rate=1)

def floor_decade(date_value):
    "Takes a date. Returns the decade."
    return (date_value.year // 10) * 10

### ACF and PACF PLots ::::::
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(t2.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(t2, lags=40, ax=ax2)

##################### 
arma_mod20 = sm.tsa.ARMA(T2, (2,0)).fit()
print(arma_mod20.params)
print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)

##################### 
arma_mod30 = sm.tsa.ARMA(T2, (3,0)).fit()
print(arma_mod30.params)

sm.stats.durbin_watson(arma_mod30.resid)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = arma_mod30.resid.plot(ax=ax);

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

### predict ::::::::
predict_sunspots = arma_mod30.predict('2014-07-14', '2014-07-23', dynamic=True)
print(predict_sunspots)
predict_sunspots
t2.tail(10)

### predict ::::::::
ax = t2.plot() (figsize=(12,8))
ax = predict_sunspots.plot(ax=ax, style='r--', label='Dynamic Prediction');
ax.legend();


arma41 = sm.tsa.ARMA(T2, (1,1)).fit()
resid = arma41.resid
## get the values 
out = arma41.fittedvalues.tolist()
out1 = pd.DataFrame(out)
out1.to_csv('F:\\Data2\\LA\\Ass2\\out1.txt')


r,q,p = sm.tsa.acf(resid, qstat=True)

data = np.c_[range(1,633), r[1:], q, p]











