# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 00:34:26 2014

@author: Saurabh
"""
import datetime
import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo
import pandas as pd
import scipy as sy
import numpy as np
import numpy.fft
import statsmodels as sm
#from scipy.fftpack import fft, ifft

t1 =pd.read_csv("F:\\Data2\\LA\\Ass2\\No_password_Fin.csv" , index_col='Date' , parse_dates=True)

t2 = t1[[4]].values
X = fft.fft(t2)

N = 633
freq = fft.fftfreq(N)
t = arange(N, dtype=float)

ind=arange(1,N/2+1)
freq[ind]
freq[-ind]

psd= abs(X[ind])**2 +abs(X[-ind])**2

figure(2)

plot(freq[ind], psd , 'k--')

where(psd>5000)

ind=where(psd>5000)[0]

freq[ind]

axvline(0.13270142)

ind_freq=arange(1,N/2+1)

ind_freq[ind]

X_cut= zeros_like(X)

X_cut[ind_freq[ind]]= X[ind_freq[ind]]
X_cut[-ind_freq[ind]]= X[-ind_freq[ind]]

X_cut

x_cut= fft.ifft(X_cut)

x_cut

plot(t, t2 , 'k,' , label='Signal')
plot(t, x_cut , 'r--' , label='Freq Cut')


plot(t, fft.ifft(X) , 'r,' , label='Signal')

x_cut

###############################
model = tsa.ARMA(t2)
import statsmodels.api as sm
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
import numpy as np
from scipy import stats
import pandas
import matplotlib.pyplot as plt

import statsmodels.api as sm

arma_mod20 = sm.tsa.ARMA(t2, (2,0)).fit()

#333333333333333333333333333333333333333333

d1 =pd.read_csv("F:\\Data2\\LA\\Ass2\\14th_OCt.csv" , index_col='num' , parse_dates=True)

d2 = d1[[0]].values
arma_mod20 = sm.tsa.ARIMA(d2, (2,1,0)).fit()

list2 = arma_mod20.fittedvalues

resid = arma_mod20.resid
r,q,p = sm.tsa.acf(resid, qstat=True)

model = sm.tsa.ARMA(d2)
result = model.fit(order=(2, 1), trend='c',method='css-mle', disp=-1)
result.params

model = sm.tsa.VAR(d2);
model.select_order(8)

test = sm.tsa.adfuller(d2)

itog = d1.describe()
d1.hist()
itog

dfx2 = pd.DataFrame(list2)
dfx2.to_csv("F:\\Data2\\LA\\Ass2\\out2.csv")


d3 = d1[[5]].values
X = fft.fft(t2)

##########################
import numpy as NP
from sklearn import datasets
from sklearn import datasets as DS
digits = DS.load_digits()
D = digits.data
T = digits.target


