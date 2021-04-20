import sys, os
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
import geopandas as gpd
from shapely.geometry import Point
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
from tqdm import tqdm
import matplotlib.dates as mdates
# from matplotlib.dates import mdates


def get_week(date, weeks):
    for week in weeks:
        s,e = week.split('_')
        if s <= date and date <= e:
            return week

def get_win(dt,win):
    return (datetime.strptime(dt,'%Y-%m-%d')-timedelta(weeks=win)).strftime('%Y-%m-%d')

fipsdf=pd.read_csv('../misc_data/US_fips_codes_names.csv')
fipsdf['FIPS']=fipsdf.FIPS.apply(lambda x: '{:05}'.format(x))
mapfips=dict(zip(fipsdf['FIPS'],fipsdf['County']+'_'+fipsdf['name']))
mapname=dict(zip(fipsdf['County']+'_'+fipsdf['name'],fipsdf['FIPS']))

goog_dates=[]
with open('input/goog_dates.txt','r') as f:
    for line in f:
        goog_dates.append(line[:-1])
goog_dates
stdt=goog_dates[-1].split('_')[0]
for i in range(1,60):
    dt0=(datetime.strptime(stdt,'%Y-%m-%d')+timedelta(weeks=i)).strftime('%Y-%m-%d')
    dt1=(datetime.strptime(dt0,'%Y-%m-%d')+timedelta(days=6)).strftime('%Y-%m-%d')
    goog_dates.append(dt0+'_'+dt1)

sorted(goog_dates)[-1]

file="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
df = pd.read_csv(file,header=0,dtype={'UID':str,'code3':str,'FIPS':str})
df=df.dropna(subset=['FIPS'])
df.loc[:,'FIPS']=df.FIPS.apply(lambda x : '{:05}'.format(int(float(x))))
all_dict={}
for st in df.Province_State:
        all_dict[st]=dict(zip(df[df.Province_State==st]['FIPS'],df[df.Province_State==st]['Admin2']))
df.drop(columns=['Country_Region','Lat','Long_','iso2','iso3','UID','code3','Admin2','Combined_Key','Province_State'],inplace=True)
agg_df=df.dropna(subset=['FIPS'])
agg_df.loc[:,'FIPS']=agg_df.FIPS.astype(float)
agg_df.FIPS=agg_df.FIPS.apply(lambda x: '{:05}'.format(int(x)))
# agg_df = df.groupby(['Province_State']).agg(np.sum)
dates = agg_df.iloc[:,1:].columns.tolist()
dates = [datetime.strptime(x, '%m/%d/%y') for x in dates]
dates = [x.strftime('%Y-%m-%d') for x in dates]
agg_df.iloc[:,1:].columns = dates
for x in agg_df.iloc[:,1:].columns:
    agg_df=agg_df.rename(columns={x:datetime.strptime(x, '%m/%d/%y').strftime('%Y-%m-%d')})
# agg_df.loc[:,'admin_st']=agg_df.Admin2+'_'+agg_df.Province_State
# agg_df=agg_df.drop(columns=['Admin2','Province_State'])
agg_df=agg_df.set_index('FIPS')
# dates = agg_df.columns
agg_df.groupby((np.arange(len(agg_df.columns)) // 7) + 1, axis=1).sum().add_prefix('s')
agg_df.columns = [get_week(x,goog_dates) for x in dates]
agg_df = agg_df.groupby(agg_df.columns,axis=1).max()
dates = agg_df.columns.values.tolist()
new_df = agg_df.copy()
for i in range(len(dates)):
    d, prev_d = dates[i], dates[i-1]
    new_df[d] = agg_df[d] - agg_df[prev_d]
    
covid_df = new_df.copy()
covid_dates = dates

# agg_df.index=agg_df.index.map(mapname)
# agg_df=agg_df[agg_df.index.notna()]
# new_df.index=new_df.index.map(mapname)
# new_df=new_df[new_df.index.notna()]

for col in agg_df.columns:
    agg_df=agg_df.rename(columns={col:col.split('_')[0]})
for col in new_df.columns:
    new_df=new_df.rename(columns={col:col.split('_')[0]})
    

for col in new_df.columns:
    new_df=new_df.rename(columns={col:datetime.strptime(col,'%Y-%m-%d').date()})

diff_df=new_df.diff(axis=1).fillna(0)
data_df=np.log(new_df.mask(new_df<=0)).fillna(0)#diff_df

new_df[new_df<=0]=1

for col in new_df.columns:
    if col=='FIPS':
        continue
    else:
        new_df=new_df.rename(columns={col:col.strftime('%Y-%m-%d')})
newdf=new_df.reset_index()
cols=new_df.columns
mnewdf=pd.melt(newdf,id_vars=['FIPS'], value_vars=cols,
        var_name='fct_date', value_name='true')
mnewdf=mnewdf.rename(columns={'FIPS':'cnty'})

dictfips=pd.read_csv('input/fips_map.csv',dtype={'FIPS':str,'St_code':str})
popdf=pd.read_csv('https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-locations/locations.csv')
popdf=popdf[['location','population']].rename(columns={'location':'FIPS'}).merge(dictfips[['FIPS','location_name','cnty']],on='FIPS')


methods=['AR_spatial','AR','ARIMA','PatchSim','PatchSim_B117','kalman','lstm']#
tdf=pd.DataFrame()
for mtd in methods:#,'PatchSim','kalman','lstm','lstm_snapshot']:#'AR_spatial_mob','mob',,'AR_spatial_mob','lstm_snapshot','AR_exog'
    tdf=tdf.append(pd.read_csv('/project/biocomplexity/aniadiga/Forecasting/covid19_ensemble/output/methods/{}/merged.csv'.format(mtd)))

tdf.cnty=pd.to_numeric(tdf.cnty)
tdf.cnty=tdf.cnty.apply(lambda x: '{:05}'.format(int(x)))

tdf=tdf.merge(mnewdf,on=['cnty','fct_date'],how='outer')
tdf=tdf.merge(popdf[['FIPS','population']].rename(columns={'FIPS':'cnty'}),on='cnty')
tdf['pk_fct_mean']=100e3*tdf['fct_mean']/tdf['population']
tdf['pk_fct_std']=100e3*tdf['fct_std']/tdf['population']
tdf['pk_true']=100e3*tdf['true']/tdf['population']

tdf['fct_rel']=tdf['fct_mean']/tdf['true']
tdf['true_rel']=tdf['true']/tdf['true']
tdf['set_type']=tdf['cnty']+'_'+tdf['method']+'_'+tdf['step_ahead']

indp=tdf[tdf.fct_mean>=tdf.population].index
tdf.loc[indp,'fct_mean']=np.nan
ind=tdf[tdf['fct_std'].isna()].index
tdf.loc[ind,'fct_std']=(tdf.loc[ind,'fct_ub']-tdf.loc[ind,'fct_lb'])/3.92
col_M=['cnty','step_ahead','method','sig']
col_W=['cnty','step_ahead','method','wts']
col_E=['cnty','horizon','fct_date','step_ahead','z_denom']
col='fct_mean'
coladj='fct_mean'#'fct_mean','pk_fct_mean'
coltrue='true'
colstd='fct_std'
col_F=['cnty','step_ahead','method',coladj]
win=4
hrzn=sys.argv[1]
#dtlist=tdf[(tdf.horizon<='2021-01-17')&(tdf.horizon>='2020-08-01')].horizon.unique()
wdf=pd.DataFrame()
for sdt in [hrzn]:#&(tdf.horizon>='2020-08-01')
    print('{} wts started'.format(sdt))
    edt=get_win(sdt,win)
    tempdf=tdf[(tdf.horizon<sdt)&(tdf.horizon>=edt)][['cnty','horizon','fct_date','step_ahead','method','set_type',coltrue,col,colstd]]
    setlist=tdf[(tdf.horizon==sdt)]['set_type'].unique()
    tempdf=tempdf[tempdf['set_type'].isin(setlist)]
    tempdf.loc[tempdf.index,'sig']=tempdf.loc[tempdf.index,colstd]
    tempdf.loc[tempdf.index,coladj]=tempdf.loc[tempdf.index,col]
    tempdf.loc[:,'wts']=1/len(tempdf.method.unique())
    # x1=[];x2=np.zeros(len(tempdf.method.unique()))
    for i in range(50):
        # E-step
#         tempdf['mse']=(tempdf[coltrue]-tempdf[col])**2
#         tempdf['gauss']=np.exp(-((tempdf[coltrue]-tempdf[col])/(np.sqrt(2)*tempdf['sig']))**2)*np.sqrt((1/(2*np.pi*tempdf['sig']**2)))
        tempdf.loc[tempdf.index,'z_num']=tempdf['wts']*np.exp(-((tempdf[coltrue]-tempdf[coladj])/(np.sqrt(2)*tempdf['sig']))**2)*np.sqrt((1/(2*np.pi*tempdf['sig']**2)))
        z_denom=tempdf.groupby(['cnty','horizon','fct_date','step_ahead'],as_index=False).sum().rename(columns={'z_num':'z_denom'})
        z_denom=z_denom[['cnty','horizon','fct_date','step_ahead','z_denom']]
        z_denom.loc[z_denom.index,'z_denom']+=1e-12
        tempdf=tempdf.merge(z_denom,on=['cnty','horizon','fct_date','step_ahead'])
    #     tempdf=tempdf.drop(['sig'],axis=1)
        tempdf.loc[tempdf.index,'z']=tempdf.loc[tempdf.index,'z_num']/tempdf.loc[tempdf.index,'z_denom']
        tempdf=tempdf.drop(['sig','z_denom','wts'],axis=1)

        # M-step
        tempdf.loc[tempdf.index,'sig_num']=tempdf.loc[tempdf.index,'z']*(tempdf[col]-tempdf[coltrue])**2
        tempdf.loc[tempdf.index,'fct_num']=tempdf.loc[tempdf.index,'z']*(tempdf[coltrue])
#         fct_temp=tempdf.groupby(['cnty','method','step_ahead'],as_index=False).sum()
#         fct_temp['fct_adj']=fct_temp['fct_num']/fct_temp['z']
        sig_temp=tempdf.groupby(['cnty','method','step_ahead'],as_index=False).sum()
#         sig_temp['sig_num']=sig_temp['sig']
        sig_temp.loc[sig_temp.index,'sig']=np.sqrt(sig_temp['sig_num'])
        w_temp=tempdf.groupby(['cnty','step_ahead','method'],as_index=False).mean().rename(columns={'z':'wts'})
#         w_temp['horizon']=sdt
#         tempdf=tempdf.drop(['wts'],axis=1)
        tempdf=tempdf.merge(w_temp[col_W],on=['cnty','method','step_ahead'],how='outer')
        tempdf=tempdf.merge(sig_temp[col_M],on=['cnty','method','step_ahead'],how='outer')
#         tempdf=tempdf.merge(fct_temp[col_F],on=['cnty','method','step_ahead'],how='outer')
    w_temp=w_temp[col_W].merge(sig_temp[col_M])
#     w_temp=w_temp.merge(fct_temp[col_F])
    w_temp['horizon']=sdt
    wdf=wdf.append(w_temp,sort=True)
#     x1=x2.copy()
#     x2=w_temp['z'].values
#     err=np.linalg.norm(x2-x1)**2

#     w_temp['horizon']=sdt
#     sig_temp['horizon']=sdt
#     w_temp=w_temp.merge(sig_temp[['cnty','step_ahead','method','sig','horizon']],on=['cnty','step_ahead','horizon','method'])
#     wdf=wdf.append(w_temp,sort=True)
#     print(i,w_temp['z'].values,err)
#     if err<=1e-6:
#         break
    # tempdf.loc[tempdf.index,'sig']=sig_temp['']
wdf['wts']=wdf['wts'].fillna(0)
twdf=wdf.groupby(['cnty','horizon','step_ahead'],as_index=False).sum()[['cnty','horizon','step_ahead','wts']]
twdf=twdf.rename(columns={'wts':'wts_sum'})
wdf=wdf.merge(twdf, on=['cnty','horizon','step_ahead'])
wdf['wts']=wdf['wts']/wdf['wts_sum']
wdf=wdf.drop(['wts_sum'],axis=1)
wdf.to_csv('output/weights/all_weeks_wts_newcomp_true_{}.csv'.format(sdt),index=None)
print('{} wts ended'.format(sdt))
# #     wdf['wts']=wdf['wts'].fillna(0)
