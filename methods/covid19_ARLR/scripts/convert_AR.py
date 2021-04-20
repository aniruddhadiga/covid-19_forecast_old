import pandas as pd
import numpy as np
import os, sys
from tqdm import tqdm
import pdb
from datetime import datetime, timedelta
ardir='/project/biocomplexity/aniadiga/Forecasting/covid19_ARLR/output/'
aroutdir='/project/biocomplexity/aniadiga/covid-19_forecast/output/'
# tqdm._instances.clear()
for mtd in ['AR_spatial','AR']:#['AR','AR_spatial','AR_spatial_mob','mob','AR_exog','AR',]:
    cnty_list=[]
    date_list=[]
    for f in os.listdir(ardir+'/'+mtd+'/'):
        if f.split('.')[-1]=='csv':
            cnty_list.append(f.split('_')[0])
            date_list.append(f.split('_')[1])
            if mtd=='AR_spatial_mob':
                fl_mtd='spatial_mob'
            else:
                fl_mtd=f.split('_')[2].split('.')[0]




    cnty_list=sorted(list(set(cnty_list)))
    date_list=sorted(list(set(date_list)))
    for dt in date_list[::-1][:6]:
        print(dt,mtd)
        tdf=pd.DataFrame()
        for cnfl in tqdm(cnty_list):
            try:
                tdf=tdf.append(pd.read_csv(ardir+'/{}/{}_{}_{}.csv'.format(mtd,cnfl,dt,fl_mtd)))
            except:
    #             print(cnfl)
                continue
        tdf=tdf.rename(columns={'cnty':'fips','horizon':'avl_date','fct_mean':'value'})
        tdf=tdf[tdf.step_ahead!='step_ahead']
        if mtd=='AR_exog':
            tdf.loc[:,'method']='AR_exog'
        tdf['horizon']=tdf.step_ahead.str.split('-',expand=True)[0].astype('int')-1
        tdf=tdf[['fips','fct_date','horizon','avl_date','method','value','fct_std']]
        sub_dt=(datetime.strptime(tdf.avl_date.unique()[0],'%Y-%m-%d')+timedelta(days=8)).strftime('%Y%m%d')
        if not os.path.exists(aroutdir+'/{}/'.format(mtd)):
            os.makedirs(aroutdir+'/{}/'.format(mtd))
        tdf.to_csv(aroutdir+'/{}/{}_{}.csv'.format(mtd,mtd,sub_dt),index=None)

