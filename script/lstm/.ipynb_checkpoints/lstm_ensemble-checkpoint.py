import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import savgol_filter
from datetime import datetime
from sklearn.cluster import KMeans
from com_func import *
from models import two_branch_lstm, lstm_mcdropout

import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--region", help="region name", required=True)
args = parser.parse_args()
region = (args.region).replace("-", " ")

print('REGION:{}'.format(region))

import json
with open('../cfg') as json_file:
    data = json.load(json_file)   
train_date = data['train_date']
skip_date = data['skip_date']
fct_date = data['fct_date']
gt_dates = data['gt_dates']
avl_date = gt_dates[gt_dates.index(train_date):]

gt_dates_short = [x.split('_')[0] for x in gt_dates]
all_dates_short = [x.split('_')[0] for x in gt_dates+fct_date]
fct_dates_short = [x.split('_')[0] for x in fct_date]

print('Preparing ground truth.')
from scipy.signal import savgol_filter
from datetime import datetime
from epiweeks import Week, Year

def get_week(date, weeks):
    for week in weeks:
        s,e = week.split('_')
        if s <= date and date <= e:
            return week
        
file = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
df = pd.read_csv(file,header=0,dtype={'UID':str,'code3':str,'FIPS':str})
df['FIPS'] = df.UID.apply(lambda x: x[3:])
df.drop(columns=['UID','Country_Region','Admin2','Lat','Long_','Province_State','iso2','iso3','code3','Combined_Key'],inplace=True)
df = df[df.FIPS!='']
agg_df = df.set_index('FIPS')
dates = agg_df.columns.tolist()
new_df = agg_df.copy()
for i in range(1,len(dates)):
    d, prev_d = dates[i], dates[i-1]
    new_df[d] = agg_df[d] - agg_df[prev_d]
    
dates = [datetime.strptime(x, '%m/%d/%y') for x in dates]
dates = [x.strftime('%Y-%m-%d') for x in dates]
new_df.columns = dates
new_df.columns = [get_week(x,gt_dates) for x in dates]
new_df = new_df.groupby(new_df.columns,axis=1).sum() 
gt = new_df.copy()
gt[gt<0] = 0.0
gt = gt.T
gt = gt.astype('float')

confirmed_df = gt.T.stack().reset_index()
confirmed_df.columns = ['cnty','date_range','confirmed']
confirmed_df['fct_date'] = confirmed_df.date_range.apply(lambda x: x.split('_')[0])

print('Prepare predictions of individual methods')
file = '/sfs/qumulo/qproject/biocomplexity/aniadiga/Forecasting/covid19_ensemble/output/ensemble/files_merged/merged_from_april.csv'
pdf = pd.read_csv(file,header=0,dtype={'cnty':str})
pdf = pdf[pdf.fct_date.isin(all_dates_short)].reset_index(drop=True)
pdf['cnty'] = pdf.cnty.apply(lambda x: x.zfill(5))

print('Prepare county list for region {}'.format(region))
state_hhs = pd.read_csv('/project/biocomplexity/lw8bn/google-mobility/PatchSim/misc_data/state_hhs_map.csv',usecols=[0,3],names=['STATE','NAME'],dtype=str)
state_hhs = state_hhs.set_index('NAME')
state_dict = state_hhs.to_dict()['STATE']
sid = state_dict[region]
pdf['st'] = pdf.cnty.apply(lambda x: x[0:2])
pdf = pdf[pdf.st==sid]
counties = np.unique(pdf.cnty.values)

steps = ['1-step_ahead','2-step_ahead','3-step_ahead','4-step_ahead']
for horizon in range(len(steps)):
    step_ahead = steps[horizon]
    print('Horizon {}'.format(step_ahead))
    pdf1 = pdf[pdf.step_ahead==step_ahead]
    pdf1 = pdf1.pivot_table(index=['cnty','fct_date'],columns='method',values='fct_mean').reset_index()
    pdf1 = pdf1.dropna(axis=0).reset_index(drop=True)

    sgt = pdf1.merge(confirmed_df,on=['cnty','fct_date'],how='left')   

    train_all = sgt.drop(['date_range'],axis=1)
#     methods = ['AR', 'ARIMA', 'AR_spatial', 'AR_spatial_mob', 'ENKF', 'PatchSim_adpt', 'lstm', 'mob', 'confirmed']
    methods = train_all.columns.values.tolist()[2:]
    train_all = train_all.pivot(index='fct_date',columns='cnty',values=methods)
    train = train_all[train_all.index.isin(gt_dates_short)]

    scaler = MinMaxScaler()
    scaler.fit(train_all.values)
    train.values[:,:] = scaler.transform(train.values)
    train_all.values[:,:] = scaler.transform(train_all.values)
    train.shape, train_all.shape

    n_back = 3
    n_ahead = 1
    n_feature = len(methods)-1
    out_feature = 1
    n_in = n_back * n_feature
    n_out = n_ahead * out_feature

    X, Y = [],[]
    counties = np.unique(train['confirmed'].columns).tolist()
    for s in counties:
        train_list = [(x,s) for x in methods[:-1]]
        values = train[train_list].values 
        reframed = series_to_supervised(values, n_back, 1)
        data = reframed.values 
        train_X = data[:, -n_in:]

        test_list = [('confirmed',s)]
        values = train[test_list].values 
        reframed = series_to_supervised(values, n_back, 1)
        data = reframed.values 
        train_Y = data[:, -n_out:]

        train_X = train_X.reshape((train_X.shape[0], n_back, n_feature))
        train_Y = train_Y.reshape(-1,1,out_feature)
        X.append(train_X)
        Y.append(train_Y)
        print('{} training data is prepared.'.format(s))

    train_X = np.concatenate(X,axis=0)
    train_Y = np.concatenate(Y,axis=0)
    print(train_X.shape, train_Y.shape)

    print('Building model...')
    model_type = 'lstm_mcdropout'
    input_shape = (n_back, n_feature)
    hidden_rnn = 32
    hidden_dense = 16
    output_dim = n_ahead
    activation = 'relu'
    model = lstm_mcdropout(input_shape, hidden_rnn, hidden_dense, output_dim, activation)
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # Model saving path
    model_name = '../model/model_lstm-{}-{}.h5'.format(region,horizon+1)
    filepath = model_name
    
    ##------if train new models-------##
    print('Preparing callbacks...')
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')

    earlystop = EarlyStopping(monitor='val_loss', 
                              patience=50, 
                              verbose=0, 
                              mode='min', 
                              restore_best_weights=True)

    callbacks = [checkpoint, earlystop]

    print('Training...')
    batch_size = 32
    epochs = 20 
    history = model.fit(train_X, train_Y, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_split=0.2, 
                        verbose=0, 
                        shuffle=False, 
                        callbacks=callbacks)
    
    ##------if load existing models-------##
    print('Loading model...')
#     model.load_weights(filepath)
    
    print('Prepare testing data...')
    counties = np.unique(train_all['confirmed'].columns).tolist()
    len_scal = len(counties)
    for k in range(len(counties)):
        s = counties[k]
        x = train_all[train_all.index<=fct_dates_short[horizon]][train_list].values[-n_back:].reshape((1, n_back, n_feature))

        print('Predicting...')
        mc_num = 50
        print('Predicting for {}...'.format(s)) 
        r,c = mc_num,5
        predict_vec = np.zeros([r,c])
        nth_scal = counties.index(s)
        predict_vec[:r,1:2] = np.array([0]*r).reshape(-1,1)
        for mc in range(mc_num):    
            pt_pd = predict_n_point_ens(model,x,scaler,len_scal,nth_scal,n_back,n_feature)
            predict_vec[mc:(mc+1),2:3] = np.array(horizon).reshape(-1,1)
            predict_vec[mc:(mc+1),3:4] = np.array([mc]).reshape(-1,1)
            predict_vec[mc:(mc+1),4:5] = np.array(pt_pd).reshape(-1,1)

        pd_df = pd.DataFrame(predict_vec, columns =['fips','date','horizon','mc','value']) 
        pd_df['fips'] = [s]*r
        if horizon==0 and k==0: header = True
        else: header = False
        pd_df.to_csv('prediction-{}.csv'.format(region),mode='a',header=header,index=False)


file = 'prediction-{}.csv'.format(region)
pdf = pd.read_csv(file,dtype={'fips':str})
pdf['avl_date'] = pdf.date.apply(lambda x: avl_date[int(x)].split('_')[0])
pdf['fct_date'] = pdf.apply(lambda x: fct_date[int(x['date']+x['horizon'])].split('_')[0],axis=1)
fct = pdf.groupby(['fips','fct_date','horizon','avl_date']).value.mean().reset_index()
fct['fct_std'] = pdf.groupby(['fips','fct_date','horizon','avl_date']).value.std().values
fct.to_csv('prediction-fmt.csv',header=False,index=False,mode='a')

import os
outfile = 'prediction-{}.csv'.format(region)
os.remove(outfile)
# os.remove('../model/model_lstm-{}-1.h5'.format(region))
# os.remove('../model/model_lstm-{}-2.h5'.format(region))
# os.remove('../model/model_lstm-{}-3.h5'.format(region))
# os.remove('../model/model_lstm-{}-4.h5'.format(region))
print('Unwanted files deleted.')
