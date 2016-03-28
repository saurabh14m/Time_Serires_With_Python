# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 19:10:31 2014

@author: Saurabh
"""

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




