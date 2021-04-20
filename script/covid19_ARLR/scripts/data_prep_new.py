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
from tqdm import tqdm
def get_week(date, weeks):
    for week in weeks:
        s,e = week.split('_')
        if s <= date and date <= e:
            return week
def outlier_rem(data):
    sd=data.std()
    mn=data.mean()
    ulim=mn+3*sd
    llim=mn-3*sd
    for i in data.index:
        if data.loc[i]<=ulim and data.loc[i]>=llim:
            data.loc[i]=data.loc[i]
        else:
#             print(i)
            data.loc[i]=np.nan
        data=data.ffill()
    return data


def get_state_data(goog_dates):
    file="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    df = pd.read_csv(file,header=0,dtype={'UID':str,'code3':str,'FIPS':str})
    df=df.groupby(['Province_State']).sum().drop(['Lat','Long_'],axis=1)
    dates = df.columns.tolist()
    dates = [datetime.strptime(x, '%m/%d/%y') for x in dates]
    dates = [x.strftime('%Y-%m-%d') for x in dates]
    for x in df.columns:
        df=df.rename(columns={x:datetime.strptime(x, '%m/%d/%y').strftime('%Y-%m-%d')})
    dates=df.columns
    df.columns = [get_week(x,goog_dates) for x in dates]
    df=df.groupby(df.columns,axis=1).max()
    dt = df.columns.values.tolist()
    new_df = df.diff(axis=1)
    
    for col in new_df.columns:
        new_df=new_df.rename(columns={col:col.split('_')[0]})
    new_df=new_df.fillna(0)
    stcddf=pd.read_csv('../input/stfips.csv',dtype={'st_code':str})
    stdict=dict(zip(stcddf['Province_State'],stcddf['st_code']))
    new_df.index=new_df.index.map(stdict)
    new_df=new_df[new_df.index.notna()]
    new_df.loc['US',:]=new_df.sum()
    with open('/project/biocomplexity/aniadiga/Forecasting/covid19_ARLR/input/st_list.txt','w') as f:
        for ln in new_df.index.unique():
                f.write("%s\n" % ln)
    return new_df        

def get_global_data(goog_dates):
    file="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    df = pd.read_csv(file,header=0,dtype={'UID':str,'code3':str,'FIPS':str})
    mapdf=pd.read_csv('https://raw.githubusercontent.com/epiforecasts/covid19-forecast-hub-europe/main/data-locations/locations_eu.csv') 
    cntry_list=mapdf.location_name.unique()#['United Kingdom','Poland','Germany']
    df=df[(df['Country/Region'].isin(cntry_list))&(df['Province/State'].isna())]
    df=df.drop(['Province/State','Lat','Long'],axis=1).set_index('Country/Region')
    dates = df.columns.tolist()
    dates = [datetime.strptime(x, '%m/%d/%y') for x in dates]
    dates = [x.strftime('%Y-%m-%d') for x in dates]
# df.columns = dates
    for x in df.columns:
            df=df.rename(columns={x:datetime.strptime(x, '%m/%d/%y').strftime('%Y-%m-%d')})
# agg_df.loc[:,'admin_st']=agg_df.Admin2+'_'+agg_df.Province_State
# agg_df=agg_df.drop(columns=['Admin2','Province_State'])
    dates=df.columns
# df=df.groupby((np.arange(len(df.columns)) // 7) + 1, axis=1).sum().add_prefix('s')
    df.columns = [get_week(x,goog_dates) for x in dates]
    df=df.groupby(df.columns,axis=1).max()
    dt = df.columns.values.tolist()
    new_df = df.diff(axis=1)
# for i in range(len(dt)):
#     d, prev_d = dt[i], dt[i-1]
#     new_df[d] = df[d] - df[prev_d]
# # covid_df = new_df.copy()
# # covid_dates = dates

# # # agg_df.index=agg_df.index.map(mapname)
# # # agg_df=agg_df[agg_df.index.notna()]
# # # new_df.index=new_df.index.map(mapname)
# # # new_df=new_df[new_df.index.notna()]
    for col in new_df.columns:
        new_df=new_df.rename(columns={col:col.split('_')[0]})
# # for col in new_df.columns:
# #     new_df=new_df.rename(columns={col:datetime.strptime(col,'%Y-%m-%d').date()})
    mapdict=dict(zip(mapdf['location_name'],mapdf['location']))
    new_df.index=new_df.index.map(mapdict)
    with open('/project/biocomplexity/aniadiga/Forecasting/covid19_ARLR/input/cntry_list.txt','w') as f:
        for ln in new_df.index.unique():
            f.write("%s\n" % ln)
    return new_df

file_mob = '/project/biocomplexity/mobility-map/proc_data/covid/us_adm2_covid_2019-2020_gadm.csv'
merge_df = pd.read_csv(file_mob,dtype={'sFIPS':str,'dFIPS':str})
dates = merge_df.date_range.unique().tolist()
goog_df = merge_df.copy()
goog_dates = dates

enddate=goog_dates[-1].split('_')[0]
for i in range(1,60):
    st_date=datetime.strptime(enddate,'%Y-%m-%d')+timedelta(days=(i)*7)
    goog_dates.append(st_date.strftime('%Y-%m-%d')+'_'+(st_date+timedelta(days=6)).strftime('%Y-%m-%d'))

fipsdf=pd.read_csv('../data/US_fips_codes_names.csv')
fipsdf['FIPS']=fipsdf.FIPS.apply(lambda x: '{:05}'.format(x))
mapfips=dict(zip(fipsdf['FIPS'],fipsdf['County']+'_'+fipsdf['name']))
mapname=dict(zip(fipsdf['County']+'_'+fipsdf['name'],fipsdf['FIPS']))
mapstcode=dict(zip(fipsdf.name.unique(),fipsdf.State_code.unique()))
cntylist=fipsdf.FIPS.values.tolist()

file="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
df = pd.read_csv(file,header=0,dtype={'UID':str,'code3':str,'FIPS':str})
df=df.dropna(subset=['FIPS'])
df.loc[:,'FIPS']=df.FIPS.apply(lambda x : '{:05}'.format(int(float(x))))
all_dict={}
for st in df.Province_State:
    all_dict[st]=dict(zip(df[df.Province_State==st]['FIPS'],df[df.Province_State==st]['Admin2']))
df.drop(columns=['Country_Region','Lat','Long_','iso2','iso3','UID','code3','Admin2','Combined_Key','Province_State'],inplace=True)
adf=df.set_index('FIPS')
adf.columns=pd.to_datetime(adf.columns,format='%m/%d/%y')
adf=adf.diff(axis=1).fillna(0)
for lc in tqdm(adf.index):
    adf.loc[lc]=outlier_rem(adf.loc[lc])
df=adf.cumsum(axis=1).reset_index()
agg_df=df.dropna(subset=['FIPS'])
agg_df.loc[:,'FIPS']=agg_df.FIPS.astype(float)
agg_df.FIPS=agg_df.FIPS.apply(lambda x: '{:05}'.format(int(x)))
# agg_df = df.groupby(['Province_State']).agg(np.sum)
dates = agg_df.iloc[:,1:].columns.tolist()
#dates = [datetime.strptime(x, '%m/%d/%y') for x in dates]
dates = [x.strftime('%Y-%m-%d') for x in dates]
agg_df.iloc[:,1:].columns = dates
for x in agg_df.iloc[:,1:].columns:
    agg_df=agg_df.rename(columns={x:x.strftime('%Y-%m-%d')})
# agg_df.loc[:,'admin_st']=agg_df.Admin2+'_'+agg_df.Province_State
# agg_df=agg_df.drop(columns=['Admin2','Province_State'])
agg_df=agg_df.set_index('FIPS')
# dates = agg_df.columns
agg_df.groupby((np.arange(len(agg_df.columns)) // 7) + 1, axis=1).sum().add_prefix('s')
agg_df.columns = [get_week(x,goog_dates) for x in dates]
agg_df = agg_df.groupby(agg_df.columns,axis=1).max()
agg_df=agg_df[agg_df.index.isin(cntylist)]
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

data_df.to_csv('../input/cnty_data.csv')


stdf=get_state_data(goog_dates)
gldf=get_global_data(goog_dates)
stdf=np.log(stdf.mask(stdf<=0)).fillna(0)
gldf=np.log(gldf.mask(gldf<=0)).fillna(0)
stdf.to_csv('../input/st_data.csv')
gldf.to_csv('../input/gl_data.csv')
### Mobility
#merge_df['flow']=pd.to_numeric(merge_df['flow'])
#merge_df.loc[:,'expflow']=np.exp(merge_df['flow'])
#outdf=merge_df.groupby(['sFIPS','date_range'],as_index=False).sum()
#outdf=outdf[outdf.sFIPS.isin(cntylist)]
#outdf['scnty']=outdf.sFIPS.apply(lambda x: mapfips[x])

#mobdf=outdf.pivot(index='scnty',columns='date_range',values='expflow')
#mobdf.index=mobdf.index.map(mapname)
## mobdf=mobdf.diff().fillna(0)
#for col in mobdf.columns:
#    mobdf=mobdf.rename(columns={col:col.split('_')[0]})
#for col in mobdf.columns:
#    mobdf=mobdf.rename(columns={col:datetime.strptime(col,'%Y-%m-%d').date()})
#mobdf=np.log(mobdf) 
#mobdf.to_csv('../input/mobility_data.csv')

###SDI
#sdidf=pd.read_csv('/project/biocomplexity/mobility-map/proc_data/covid/cnty_sdm_updt.csv')
#sdidf=sdidf.drop('Unnamed: 0',axis=1).set_index('FIPS')
#for col in sdidf.columns:                                                                                   
#    sdidf=sdidf.rename(columns={col:col.split('_')[0]})                                                     
#for col in sdidf.columns:                                                                                   
#    sdidf=sdidf.rename(columns={col:datetime.strptime(col,'%Y-%m-%d').date()}) 
#dsdidf=((sdidf-sdidf.shift(axis=1))/sdidf.shift(axis=1))*100                             
#sdidf.to_csv('../input/sdi_data.csv') 
#dsdidf.to_csv('../input/rate_sdi_data.csv')

###Doctor's visit 
#dfile='/project/biocomplexity/COVID-19_commons/data/COVIDCast/doctor-visits_county.csv'
#ddf=pd.read_csv(dfile)
#ddf['geo_value']=ddf.geo_value.apply(lambda x: '{:05}'.format(x))
#ddf=ddf.merge(fipsdf.rename(columns={'FIPS':'geo_value'}),on='geo_value',how='left')
#pddf=ddf.pivot(index='geo_value',columns='time_value',values='value')
#rem_cnty_list=list(set(fipsdf.FIPS.unique())-set(pddf.index))
#tempdf=pd.DataFrame(index=rem_cnty_list,columns=pddf.columns)
#pddf=pddf.append(tempdf)
#pddf=pddf.fillna(method='ffill',axis=1)
#pddf=pddf.fillna(0,axis=1)
#pddf.columns=pddf.columns.astype('datetime64[ns]')
#pddf=pddf.resample('W-Sun',axis=1).mean()
#pddf.columns=pddf.columns.astype('str')
#pddf.to_csv('../input/doc_visit_data.csv')
with open('../input/statewise_list.txt','w') as ffile:

    for st in mapstcode.keys():
        st_list=list(all_dict[st].keys())
        filename='../input/{}_cnty_list.txt'.format(mapstcode[st])
        ffile.write('{}\n'.format(filename))
        with open(filename,'w') as f:
            for ln in st_list:
                f.write('{}\n'.format(ln))
        f.close()
        st_data_df=data_df[data_df.index.isin(st_list)]
        st_data_df.to_csv('../input/{}_cnty_data.csv'.format(mapstcode[st]))
ffile.close()
#va_list=list(all_dict['Virginia'].keys())
#with open('../input/va_cnty_list.txt','w') as f:
#    for ln in va_list:
#        f.write('{}\n'.format(ln))
#f.close()
#va_data_df=data_df[data_df.index.isin(va_list)]
#va_data_df.to_csv('../input/va_cnty_data.csv')
#pdb.set_trace()

