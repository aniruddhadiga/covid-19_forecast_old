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

def get_week(date, weeks):
    for week in weeks:
        s,e = week.split('_')
        if s <= date and date <= e:
            return week

file = '/project/biocomplexity/mobility-map/proc_data/covid/us_adm2_covid_2019-2020_gadm.csv'
merge_df = pd.read_csv(file,dtype={'sFIPS':str,'dFIPS':str})
dates = merge_df.date_range.unique().tolist()
goog_df = merge_df.copy()
goog_dates = dates

enddate=goog_dates[-1].split('_')[0]
for i in range(1,6):
    st_date=datetime.strptime(enddate,'%Y-%m-%d')+timedelta(days=(i)*7)
    goog_dates.append(st_date.strftime('%Y-%m-%d')+'_'+(st_date+timedelta(days=6)).strftime('%Y-%m-%d'))

fipsdf=pd.read_csv('../data/US_fips_codes_names.csv')
fipsdf['FIPS']=fipsdf.FIPS.apply(lambda x: '{:05}'.format(x))
mapfips=dict(zip(fipsdf['FIPS'],fipsdf['County']+'_'+fipsdf['name']))
mapname=dict(zip(fipsdf['County']+'_'+fipsdf['name'],fipsdf['FIPS']))

file="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
df = pd.read_csv(file,header=0,dtype={'UID':str,'code3':str,'FIPS':str})
df.drop(columns=['Country_Region','Lat','Long_','iso2','iso3','UID','code3','FIPS','Combined_Key'],inplace=True)
agg_df=df.dropna(subset=['Admin2'])
# agg_df = df.groupby(['Province_State']).agg(np.sum)
dates = agg_df.iloc[:,2:].columns.tolist()
dates = [datetime.strptime(x, '%m/%d/%y') for x in dates]
dates = [x.strftime('%Y-%m-%d') for x in dates]
agg_df.iloc[:,2:].columns = dates
for x in agg_df.iloc[:,2:].columns:
    agg_df=agg_df.rename(columns={x:datetime.strptime(x, '%m/%d/%y').strftime('%Y-%m-%d')})
agg_df.loc[:,'admin_st']=agg_df.Admin2+'_'+agg_df.Province_State
agg_df=agg_df.drop(columns=['Admin2','Province_State'])
agg_df=agg_df.set_index('admin_st')
# dates = agg_df.columns
agg_df.groupby((np.arange(len(agg_df.columns)) // 7) + 1, axis=1).sum().add_prefix('s')
agg_df.columns = [get_week(x,goog_dates) for x in dates]
agg_df = agg_df.groupby(agg_df.columns,axis=1).max()
dates = agg_df.columns.values.tolist()
new_df = agg_df.copy()
for i in range(1,len(dates)):
    d, prev_d = dates[i], dates[i-1]
    new_df[d] = agg_df[d] - agg_df[prev_d]

covid_df = new_df.copy()
covid_dates = dates

agg_df.index=agg_df.index.map(mapname)
agg_df=agg_df[agg_df.index.notna()]
new_df.index=new_df.index.map(mapname)
new_df=new_df[new_df.index.notna()]

for col in agg_df.columns:
    agg_df=agg_df.rename(columns={col:col.split('_')[0]})
for col in new_df.columns:
    new_df=new_df.rename(columns={col:col.split('_')[0]})
for ind in new_df.index:
        new_df.loc[ind,:]=np.abs(np.ceil(savgol_filter(new_df.loc[ind,:],axis=0,window_length=5, polyorder=2)))
diff_df=new_df.diff(axis=1).fillna(0)
data_df=np.log(new_df.mask(new_df<=0)).fillna(0)#diff_df



## Mobility
merge_df.loc[:,'expflow']=np.exp(merge_df['flow'])
outdf=merge_df.groupby(['sFIPS','date_range'],as_index=False).sum()
outdf['scnty']=outdf.sFIPS.apply(lambda x: mapfips[x])

mobdf=outdf.pivot(index='scnty',columns='date_range',values='expflow')
mobdf.index=mobdf.index.map(mapname)
# mobdf=mobdf.diff().fillna(0)
for col in mobdf.columns:
    mobdf=mobdf.rename(columns={col:col.split('_')[0]})
for col in mobdf.columns:
    mobdf=mobdf.rename(columns={col:datetime.strptime(col,'%Y-%m-%d').date()})
mobdf=np.log(mobdf)    
for col in new_df.columns:
    new_df=new_df.rename(columns={col:datetime.strptime(col,'%Y-%m-%d').date()})
    
for col in diff_df.columns:
    data_df=data_df.rename(columns={col:datetime.strptime(col,'%Y-%m-%d').date()})

data_df.to_csv('../input/cnty_data.csv')
mobdf.to_csv('../input/mobility_data.csv')

va_list=list(all_dict['Virginia'].keys())
with open('../input/va_cnty_list.txt','w') as f:
    for ln in va_list:
        f.write('{}\n'.format(ln))
f.close()
