# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 23:19:04 2014

@author: Saurabh
"""

import datetime
import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo
#import scikits.timeseries as ts
#import scikits.timeseries.lib.plotlib as tpl
import pandas as pd
import scipy as sy
import numpy as np
from scipy.fftpack import fft, ifft

t1 =pd.read_csv("F:\\Data2\\LA\\Ass2\\No_password_Fin.csv" , index_col='Date' , parse_dates=True)
t2 = t1[[1]].values
X = scipy.fft(t2)

Xinv = ifft(X)
np.sum(t2)

N = 633
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
 y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
 
import numpy as np
def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

np.allclose(DFT_slow(t2), np.fft.fft(t2))

%timeit DFT_slow(t2)
%timeit np.fft.fft(t2)
#################################33
import scipy.fftpack as syfp
import pylab as pyl
array  = t2.copy()
length = len(array)
# Create time data for x axis based on array length
x = sy.linspace(0.00001, length*0.00001, num=length)

# Do FFT analysis of array
FFT = sy.fft(array)
# Getting the related frequencies
freqs = syfp.fftfreq(array.size, d=(x[1]-x[0]))

# Create subplot windows and show plot
pyl.subplot(211)
pyl.plot(x, array)
pyl.subplot(212)
pyl.plot(freqs, sy.log10(FFT), 'x')
pyl.show()

import matplotlib.pylab as plt
import matplotlib.mlab as mlb

Fs = 1./(d[1]- d[0])  # sampling frequency
plt.psd(array, Fs=Fs, detrend=mlb.detrend_mean) 
plt.show()

plot(X, 'b-')

dataF = np.abs(np.fft.fftshift(np.fft.fft(t2)))
plot(dataF, 'b-')

