import numpy as np
from math import sqrt
import pandas as pd
import datetime
from filterpy.kalman import EnsembleKalmanFilter as EnKF
from filterpy.kalman import KalmanFilter
np.random.seed(1234)

def create_ts(df):
  ts=df
  ts = df.drop(['UID', 'iso2', 'iso3', 'code3', 'Combined_Key','Admin2', 'Province_State', 'Country_Region', 'Lat', 'Long_'], axis=1)
  ts['FIPS'] = ts['FIPS'].fillna('')
  ts = ts[~(ts['FIPS']=='')].reset_index(drop=True)
  #ts['FIPS'] = ts['FIPS'].astype(int).astype('str')
  ts = ts.rename({'FIPS': 'region'}, axis='columns') 
  ts.set_index('region')
  ts=ts.T
  ts.columns=ts.loc['region']
  ts=ts.drop('region')
  ts=ts.fillna(0)
  return (ts)

url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
jhupull = pd.read_csv(url, error_bad_lines=False)

##nlist = [40]
##for nval in nlist:
##  print(nval)
##week_minus = [12,11,10,9,8,7,4,3,2,1]
##for wkmn in week_minus:
##  print(wkmn)
main_confirmed = jhupull
state_list = list( sorted(set(main_confirmed['Province_State'])-{'Northern Mariana Islands', 'American Samoa', 'Grand Princess', 'Guam','Virgin Islands','Diamond Princess','District of Columbia'})) #Get all states

satcols = []
for i in main_confirmed.columns[11:]:  ## removed last week for eval now
  col_name = i.split('/')
  if datetime.date(int(col_name[2]),int(col_name[0]),int(col_name[1])).weekday() == 5 : satcols.append(i) #Monday 0, Sunday 6
main_confirmed = main_confirmed[main_confirmed.columns[:11].tolist()+satcols]
main_confirmed = main_confirmed[main_confirmed['FIPS']<80000]
main_confirmed = main_confirmed[~main_confirmed['Province_State'].isin(['Northern Mariana Islands', 'American Samoa', 'Grand Princess', 'Guam','Virgin Islands','Diamond Princess','District of Columbia'])].reset_index(drop=True)

#main_confirmed = main_confirmed[main_confirmed.columns[:-wkmn]]
#nval = len(main_confirmed.columns)-11+4 ## give n val as per obs in data + 4 week
nval = 100

confirmed=main_confirmed
ts_backup = create_ts(confirmed)
ts = ts_backup

ts_r=ts.reset_index()
ts_r=ts_r.rename(columns = {'index':'date'})
ts_r['date']=pd.to_datetime(ts_r['date'] ,errors ='coerce')

final_list = []
final_sdlst = []

def hx(x):
    return np.array([x[0]])
def fx(x, dt):
    return np.dot(F, x)

for echcnty in ts_r.columns[1:]:
    listz = ts_r[echcnty].tolist()

    F = np.array([[1., 1.],[0., 1.]])
    x = np.array([0., 1.])
    P = np.eye(2) * 100.
    enf = EnKF(x=x, P=P, dim_z=1, dt=1., N=nval, hx=hx, fx=fx)

    fin_res = []
    fin_sd = []
    for z in listz:
        enf.predict()
        enf.update(np.asarray([z]))
        # # save data
        # results.append(enf.x[0])
        # ps.append(3*(enf.P[0,0]**.5))
    # results = np.asarray(results)
    # ps = np.asarray(ps)
    for i in range(0,4):
      enf.predict()
      fin_res.append(enf.x[0])
      fin_sd.append((enf.P[0,0]**.5))

    final_list.append([echcnty]+fin_res)
    final_sdlst.append([echcnty]+fin_sd)

lastday = ts_r['date'].iloc[-1].date()
fnl2 = pd.DataFrame(final_list, columns=['FIPS', lastday+datetime.timedelta(days=7), lastday+datetime.timedelta(days=14), lastday+datetime.timedelta(days=21), lastday+datetime.timedelta(days=28)])
fnl2.insert(1,lastday,ts_r.iloc[-1][1:].tolist())

sddf = pd.DataFrame(final_sdlst)

wksum = pd.DataFrame()
wksum['FIPS'] = fnl2['FIPS']
wksum[fnl2.columns[1]+datetime.timedelta(1)] = (fnl2[fnl2.columns[2]]-fnl2[fnl2.columns[1]])
wksum[fnl2.columns[2]+datetime.timedelta(1)] = (fnl2[fnl2.columns[3]]-fnl2[fnl2.columns[2]])
wksum[fnl2.columns[3]+datetime.timedelta(1)] = (fnl2[fnl2.columns[4]]-fnl2[fnl2.columns[3]])
wksum[fnl2.columns[4]+datetime.timedelta(1)] = (fnl2[fnl2.columns[5]]-fnl2[fnl2.columns[4]])

##pblmdf = wksum[(abs(wksum[wksum.columns[1]] - wksum[wksum.columns[2]])>2000) | (wksum[wksum.columns[1]] <0)| (wksum[wksum.columns[2]] <0)| (wksum[wksum.columns[3]] <0)| (wksum[wksum.columns[4]] <0)]
##mainpull = main_confirmed[main_confirmed['FIPS'].isin(pblmdf['FIPS'])]
##mpullst = mainpull[mainpull.columns[-1]]-mainpull[mainpull.columns[-2]].tolist()
##pblmdf['crction'] = [float(max(0,i)) for i in mpullst]
##pblmdf

wksum[wksum < 0] = 0

avldt = wksum.columns[1]-datetime.timedelta(7) #'2020-09-27'
spreadlst = []
for i in range(wksum.shape[0]):
    spreadlst.append([wksum['FIPS'].iloc[i], wksum.columns[1], 0, avldt, wksum[wksum.columns[1]].iloc[i], sddf[sddf.columns[1]].iloc[i]])
    spreadlst.append([wksum['FIPS'].iloc[i], wksum.columns[2], 1, avldt, wksum[wksum.columns[2]].iloc[i], sddf[sddf.columns[2]].iloc[i]])
    spreadlst.append([wksum['FIPS'].iloc[i], wksum.columns[3], 2, avldt, wksum[wksum.columns[3]].iloc[i], sddf[sddf.columns[3]].iloc[i]])
    spreadlst.append([wksum['FIPS'].iloc[i], wksum.columns[4], 3, avldt, wksum[wksum.columns[4]].iloc[i], sddf[sddf.columns[4]].iloc[i]])
spreaddf = pd.DataFrame(spreadlst, columns=['fips','fct_date','horizon','avl_date','value','fct_std'])

spreaddf = spreaddf.sort_values(['fct_date','fips']).reset_index(drop=True)
spreaddf['fips'] = spreaddf['fips'].astype(int)

spreaddf.to_csv('enkf_weekly_cu_n'+str(nval)+'_'+str(wksum.columns[1]).replace('-','')+'.csv')
