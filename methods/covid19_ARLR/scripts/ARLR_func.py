import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from joblib import Parallel, delayed
import multiprocessing
from scipy.stats import entropy
from datetime import datetime, timedelta
import epiweeks as epi
from scipy.stats import pearsonr
from scipy.signal import correlate
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fftpack import fft, ifft
from scipy.optimize import nnls
import scipy.stats as stats
from scipy.optimize import leastsq
from scipy.signal import savgol_filter
from collections import defaultdict
import statsmodels.api as sm

def gen_noise(y,tol,trfn):
    if trfn=='log':
        yn=np.exp(y)
#    for i in range(len(yn)):
#        try:
#            yn[i]=np.random.randint(yn[i]-.1,yn[i]+.1,dtype=np.int64)#yn[i]-(yn[i]*.01) yn[i]+(yn[i]*0.01)
#        except:
#            continue
    if trfn=='log':
        yn=np.log(yn.mask(yn<=0)).fillna(0)
    return yn

def get_mape(y_pred,y_true):
    mape=np.mean(np.abs(y_pred-y_true)/y_true)*100
    return mape

def get_mse(y_pred,y_true):
    mse=np.mean((y_pred-y_true)**2)
    return mse

