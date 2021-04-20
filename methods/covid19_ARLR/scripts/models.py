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
from ARLR_func import gen_noise, get_mape, get_mse

def gen_tup(dict_df,cnty_name,shift_range,step):
    tup_tup=[]
    for key,values in dict_df.items():
        if key == 'cases':
            range_ind=range(step,shift_range+step)
            for ind in list(values.index):
#                 if ind!=cnty_name:
#                     continue
# #                     range_ind=range(1,shift_range)
#                 else:
# #                     range_ind=range(1,shift_range)
                for sh in range_ind:
                    tup_tup.append((key,ind,sh))
        else:
            for sh in range(step-1,shift_range):
                tup_tup.append((key,cnty_name,sh))
    return tup_tup

def gen_tup_mob(dict_df,cnty_name,shift_range,step):
    tup_tup=[]
    for key,values in dict_df.items():
        if key == 'cases':
            continue
        elif key=='mob':
            for sh in range(step-1,shift_range):
                tup_tup.append((key,cnty_name,sh))
    return tup_tup

def gen_tup_ar(dict_df,cnty_name,shift_range,step):
    tup_tup=[]
    for key,values in dict_df.items():
        if key == 'cases':
            for ind in list(values.index):
                if ind!=cnty_name:
                    continue
# #                     range_ind=range(1,shift_range)
                else:
                    range_ind=range(step,shift_range+step)
                    for sh in range_ind:
                        tup_tup.append((key,ind,sh))
        else:
            continue
    return tup_tup

def forward_sel_solver(data,target):
    initial_features = data.columns.tolist()
    best_features = []
    significance_level=0.05
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()#
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            model = sm.OLS(target, sm.add_constant(data[best_features])).fit()
            break
    return model, best_features


def predictor_spatial_mob(data_df,mobdf,cnty_list,hrzn_date_str,step_ahead):
    cntyresdf=pd.DataFrame(columns=['method','cnty','horizon','fct_date','fct','true'],index=None)
    for hrzn in hrzn_date_str:
        for cntys in cnty_list:#[data_df.index[0]]:
            for noise_iter in range(1):
                print(hrzn,cntys,noise_iter)
                ndata_df=data_df.copy(deep=True)
                for cnty in data_df.index:
                    ndata_df.loc[cnty,:]=gen_noise(data_df.loc[cnty,:],0,'log')
                
                dict_df={'cases':ndata_df, 'mob':mobdf}
                cnty_name=cntys#'51059'
                shift_fac=3
                for mul_step in range(1,step_ahead+1):
                    tup_tup_new=gen_tup(dict_df,cnty_name,shift_fac,mul_step)

        #             hrzn_date_str=
                    hrzn_date=datetime.strptime(hrzn,'%Y-%m-%d').date()
                    pred_len=len(tup_tup_new)
                    win=8
                    A=np.zeros([win,pred_len])
                    ii=0
                    for tup in tup_tup_new:
                        tempdf=dict_df[tup[0]]
                        tempdf=tempdf.loc[:,tempdf.columns<=hrzn_date]#+timedelta(weeks=1)
                        try:
                            A[:,ii]=tempdf.loc[tup[1],:].shift(tup[2]).fillna(0).values[-win:]
                        except:
                            continue
                            #pdb.set_trace()
                        ii+=1

                    y=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date][-win:]#+timedelta(weeks=1)

                    cols=list(tup_tup_new)
                    idx=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date][-win:].index#+timedelta(weeks=1)
                    Adf=pd.DataFrame(data=A,columns=cols,index=idx)

                    cols=list(tup_tup_new)
                    idx=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date+timedelta(weeks=mul_step)][-win:].index#+timedelta(weeks=1)
                    yf=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date+timedelta(weeks=mul_step)][-win:]#+timedelta(weeks=1)

                    # train model
                    target=y
                    data=Adf
                    model, best_features=forward_sel_solver(data,target)
                    y_pred=model.predict()
                    y_pred=np.exp(y_pred)
                    y_true=target.values
                    y_true=np.exp(y_true)
                    train_err_mape=get_mape(y_pred,y_true)
                    train_err_mse=get_mse(y_pred,y_true)
                    # forecast
                    nc_idx=model.params.index[1:]
                    coeffs=model.params.values+0.1*np.random.randn(1000,len(model.params))
                    yfh=np.matmul(coeffs[:,1:],Adf.loc[hrzn_date,model.params.index[1:]].values)+coeffs[:,0]
                    fct_week=hrzn_date+timedelta(weeks=mul_step)
                    # y[hrzn_date+timedelta(weeks=1)]
            #         print('{}-step ahead: data available till the {}'.format(mul_step,hrzn_date))
            #         try:
            #             print('forecast for week {}: yfh={}, y={}'.format(fct_week,yfh,yf[fct_week]))
            #             print('forecast for week {}: casesf={}, cases={}'.format(fct_week,np.exp(yfh),np.exp(yf[fct_week])))
            #             print('MAPE={}'.format(mape_met))
            #         except:
            #             print('forecast for week {}: yfh={}, y={}'.format(fct_week,yfh,np.nan))
            #             print('forecast for week {}: casesf={}, cases={}'.format(fct_week,np.exp(yfh),np.nan))

                    # store result
                    resdf=pd.DataFrame(columns=['method','cnty','horizon','fct_date','step_ahead','iter','fct','true','fct_mape','train_mape','train_mse'],index=None)
                    resdf.loc[0,'method']='spatial_mob'
                    resdf.loc[0,'cnty']=cnty_name
                    resdf.loc[0,'horizon']=hrzn_date
                    resdf.loc[0,'fct_date']=fct_week
                    resdf.loc[0,'fct_mean']=np.exp(yfh).mean()
                    resdf.loc[0,'fct_std']=np.exp(yfh).std()
                    resdf.loc[0,'step_ahead']='{}-step_ahead'.format(mul_step)
                    resdf.loc[0,'iter']=noise_iter
                    resdf.loc[0,'train_mape']=train_err_mape
                    resdf.loc[0,'train_mse']=train_err_mse

                    try:
                        resdf.loc[0,'true']=np.exp(yf[fct_week])
                        mape_met=100*np.abs((np.exp(yfh)-np.exp(yf[fct_week]))/np.exp(yf[fct_week]))
                        resdf.loc[0,'fct_mape']=mape_met

                    except:
                        resdf.loc[0,'true']=np.nan
                        resdf.loc[0,'fct_mape']=np.nan
                    cntyresdf=cntyresdf.append(resdf,ignore_index=True)
    
    outdir='../output/AR_spatial_mob/'
    if not os.path.exists(outdir):
            os.makedirs(outdir)
    
    outfile=outdir+cnty_name+'_'+hrzn+'_spatial_mob.csv'
    cntyresdf.to_csv(outfile,index=None,mode='w')
    return cntyresdf


def predictor_spatial(data_df,cnty_list,hrzn_date_str,step_ahead,outdir='../output/AR_spatial/',pkldir='../pkl/AR_spatial/'):
    if not os.path.exists(outdir):
            os.makedirs(outdir)
    
    if not os.path.exists(pkldir):
            os.makedirs(pkldir)
    for hrzn in hrzn_date_str:

        cntyresdf=pd.DataFrame(columns=['method','cnty','horizon','fct_date','fct','true'],index=None)
        for cntys in cnty_list:#[data_df.index[0]]:
            for noise_iter in range(1):
                print(hrzn,cntys,noise_iter)
                ndata_df=data_df.copy(deep=True)
                for cnty in data_df.index:
                    ndata_df.loc[cnty,:]=gen_noise(data_df.loc[cnty,:],0,'log')

                dict_df={'cases':ndata_df}
                cnty_name=cntys#'51059'
                shift_fac=3
                for mul_step in range(1,step_ahead+1):
                    tup_tup_new=gen_tup(dict_df,cnty_name,shift_fac,mul_step)
                    hrzn_date=datetime.strptime(hrzn,'%Y-%m-%d').date()
                    pred_len=len(tup_tup_new)
                    win=8
                    A=np.zeros([win,pred_len])
                    ii=0

                    for tup in tup_tup_new:
                        tempdf=dict_df[tup[0]]
                        tempdf=tempdf.loc[:,tempdf.columns<=hrzn_date]#+timedelta(weeks=1)
                        try:
                            A[:,ii]=tempdf.loc[tup[1],:].shift(tup[2]).fillna(0).values[-win:]
                        except:
                            continue
                        ii+=1

                    y=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date][-win:]#+timedelta(weeks=1)

                    cols=list(tup_tup_new)
                    idx=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date][-win:].index#+timedelta(weeks=1)
                    Adf=pd.DataFrame(data=A,columns=cols,index=idx)

                    cols=list(tup_tup_new)
                    idx=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date+timedelta(weeks=mul_step)][-win:].index#+timedelta(weeks=1)
                    yf=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date+timedelta(weeks=mul_step)][-win:]#+timedelta(weeks=1)
                    # train model
                    target=y
                    data=Adf
                    model, best_features=forward_sel_solver(data,target)
                    y_pred=model.predict()
                    #y_pred=np.exp(y_pred)
                    y_true=target.values
                    #y_true=np.exp(y_true)
                    train_err_mape=get_mape(y_pred,y_true)
                    train_err_mse=get_mse(y_pred,y_true)
                    # forecast
                    nc_idx=model.params.index[1:]
                    coeffs=model.params.values+0.01*np.random.randn(1000,len(model.params))
                    yfh=np.matmul(coeffs[:,1:],Adf.loc[hrzn_date,model.params.index[1:]].values)+coeffs[:,0]
                    fct_week=hrzn_date+timedelta(weeks=mul_step)
                    #ylb=yfh.mean()-2*np.sqrt(train_err_mse)
                    #yub=yfh.mean()+2*np.sqrt(train_err_mse)
                    ylb=yfh.mean()-2*yfh.std()
                    yub=yfh.mean()+2*yfh.std()
                    # y[hrzn_date+timedelta(weeks=1)]
            #         print('{}-step ahead: data available till the {}'.format(mul_step,hrzn_date))
            #         try:
            #             print('forecast for week {}: yfh={}, y={}'.format(fct_week,yfh,yf[fct_week]))
            #             print('forecast for week {}: casesf={}, cases={}'.format(fct_week,np.exp(yfh),np.exp(yf[fct_week])))
            #             print('MAPE={}'.format(mape_met))
            #         except:
            #             print('forecast for week {}: yfh={}, y={}'.format(fct_week,yfh,np.nan))
            #             print('forecast for week {}: casesf={}, cases={}'.format(fct_week,np.exp(yfh),np.nan))

                    # store result
                    resdf=pd.DataFrame(columns=['method','cnty','horizon','fct_date','step_ahead','iter','fct','true','fct_mape','train_mape','train_mse'],index=None)
                    resdf.loc[0,'method']='spatial'
                    resdf.loc[0,'cnty']=cnty_name
                    resdf.loc[0,'horizon']=hrzn_date
                    resdf.loc[0,'fct_date']=fct_week
                    resdf.loc[0,'fct_mean']=np.exp(yfh.mean())
                    resdf.loc[0,'fct_std']=np.exp(yfh).std()
                    resdf.loc[0,'fct_lb']=np.exp(ylb)
                    resdf.loc[0,'fct_ub']=np.exp(yub)

                    resdf.loc[0,'step_ahead']='{}-step_ahead'.format(mul_step)
                    resdf.loc[0,'iter']=noise_iter
                    resdf.loc[0,'train_mape']=train_err_mape
                    resdf.loc[0,'train_mse']=train_err_mse

                    try:
                        resdf.loc[0,'true']=np.exp(yf[fct_week])
                        mape_met=100*np.abs((np.exp(yfh)-np.exp(yf[fct_week]))/np.exp(yf[fct_week]))
                        resdf.loc[0,'fct_mape']=mape_met

                    except:
                        resdf.loc[0,'true']=np.nan
                        resdf.loc[0,'fct_mape']=np.nan
                    cntyresdf=cntyresdf.append(resdf,ignore_index=True)
                    pklfile=pkldir+cnty_name+'_'+hrzn+'_'+str(mul_step)+'-step_spatial.pkl'
                    model.save(pklfile)
                    outfile=outdir+cnty_name+'_'+hrzn+'_spatial.csv'
                cntyresdf.to_csv(outfile,index=None,mode='w')

    return cntyresdf

def predictor_ar(data_df,mobdf,cnty_list,hrzn_date_str,step_ahead,outdir='../output/AR/',pkldir='../pkl/AR/'):
    #outdir='../output/AR/'
    if not os.path.exists(outdir):
            os.makedirs(outdir)
    #pkldir='../pkl/AR/'
    if not os.path.exists(pkldir):
            os.makedirs(pkldir)


    for hrzn in hrzn_date_str:
        cntyresdf=pd.DataFrame(columns=['method','cnty','horizon','fct_date','fct','true'],index=None)
        for cntys in cnty_list:
            for noise_iter in range(1):
                print(hrzn,cntys,noise_iter)
                ndata_df=data_df.copy(deep=True)
                for cnty in data_df.index:
                    ndata_df.loc[cnty,:]=gen_noise(data_df.loc[cnty,:],0,'log')
                dict_df={'cases':ndata_df, 'mob':mobdf}
                cnty_name=cntys#'51059'
                shift_fac=7
                for mul_step in range(1,step_ahead+1):
                    tup_tup_new=gen_tup_ar(dict_df,cnty_name,shift_fac,mul_step)
        #             hrzn_date_str=
                    #hrzn_date=hrzn_date_str
                    hrzn_date=datetime.strptime(hrzn,'%Y-%m-%d').date()
                    pred_len=len(tup_tup_new)
                    win=8
                    A=np.zeros([win,pred_len])
                    ii=0
                    for tup in tup_tup_new:
                        tempdf=dict_df[tup[0]]
                        tempdf=tempdf.loc[:,tempdf.columns<=hrzn_date]#+timedelta(weeks=1)
                        try:
                            A[:,ii]=tempdf.loc[tup[1],:].shift(tup[2]).fillna(0).values[-win:]
                        except:
                            continue
                        ii+=1

                    y=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date][-win:]#+timedelta(weeks=1)

                    cols=list(tup_tup_new)
                    idx=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date][-win:].index#+timedelta(weeks=1)
                    Adf=pd.DataFrame(data=A,columns=cols,index=idx)

                    cols=list(tup_tup_new)
                    idx=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date+timedelta(weeks=mul_step)][-win:].index#+timedelta(weeks=1)
                    yf=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date+timedelta(weeks=mul_step)][-win:]#+timedelta(weeks=1)

                    # train model
                    target=y
                    data=Adf
                    try:
                        model, best_features=forward_sel_solver(data,target)
                    except:
                        continue
                        #pdb.set_trace()
                    y_pred=model.predict()
                    y_pred=y_pred
                    y_true=target.values
                    y_true=y_true
                    train_err_mape=get_mape(y_pred,y_true)
                    train_err_mse=get_mse(y_pred,y_true)
                    
                    # forecast
                    nc_idx=model.params.index[1:]
                    coeffs=model.params.values+0.01*np.random.randn(1000,len(model.params))
                    yfh=np.matmul(coeffs[:,1:],Adf.loc[hrzn_date,model.params.index[1:]].values)+coeffs[:,0]
                    fct_week=hrzn_date+timedelta(weeks=mul_step)
                    #ylb=yfh.mean()-2*np.sqrt(train_err_mse)
                    #yub=yfh.mean()+2*np.sqrt(train_err_mse)
                    ylb=yfh.mean()-2*yfh.std()
                    yub=yfh.mean()+2*yfh.std()

                    # y[hrzn_date+timedelta(weeks=1)]
            #         print('{}-step ahead: data available till the {}'.format(mul_step,hrzn_date))
            #         try:
            #             print('forecast for week {}: yfh={}, y={}'.format(fct_week,yfh,yf[fct_week]))
            #             print('forecast for week {}: casesf={}, cases={}'.format(fct_week,np.exp(yfh),np.exp(yf[fct_week])))
            #             print('MAPE={}'.format(mape_met))
            #         except:
            #             print('forecast for week {}: yfh={}, y={}'.format(fct_week,yfh,np.nan))
            #             print('forecast for week {}: casesf={}, cases={}'.format(fct_week,np.exp(yfh),np.nan))

                    # store result
                    resdf=pd.DataFrame(columns=['method','cnty','horizon','fct_date','step_ahead','iter','fct','true','fct_mape','train_mape','train_mse'],index=None)
                    resdf.loc[0,'method']='ar'
                    resdf.loc[0,'cnty']=cnty_name
                    resdf.loc[0,'horizon']=hrzn_date
                    resdf.loc[0,'fct_date']=fct_week
                    resdf.loc[0,'fct_mean']=np.exp(yfh).mean()
                    resdf.loc[0,'fct_std']=np.exp(yfh).std()
                    resdf.loc[0,'fct_lb']=np.exp(ylb)
                    resdf.loc[0,'fct_ub']=np.exp(yub)
                    resdf.loc[0,'step_ahead']='{}-step_ahead'.format(mul_step)
                    resdf.loc[0,'iter']=noise_iter
                    resdf.loc[0,'train_mape']=train_err_mape
                    resdf.loc[0,'train_mse']=train_err_mse

                    try:
                        resdf.loc[0,'true']=np.exp(yf[fct_week])
                        mape_met=100*np.abs((np.exp(yfh)-np.exp(yf[fct_week]))/np.exp(yf[fct_week]))
                        resdf.loc[0,'fct_mape']=mape_met

                    except:
                        resdf.loc[0,'true']=np.nan
                        resdf.loc[0,'fct_mape']=np.nan
                    cntyresdf=cntyresdf.append(resdf,ignore_index=True)
                    pklfile=pkldir+cnty_name+'_'+hrzn+'_'+str(mul_step)+'-step_ar.pkl'
                    model.save(pklfile)
                    outfile=outdir+cnty_name+'_'+hrzn+'_ar.csv'
                cntyresdf.to_csv(outfile,index=None,mode='w')

    return cntyresdf

def predictor_mob(data_df,mobdf,cnty_list,hrzn_date_str,step_ahead):
    cntyresdf=pd.DataFrame(columns=['method','cnty','horizon','fct_date','fct','true'],index=None)
    for hrzn in hrzn_date_str:
        for cntys in cnty_list:#[data_df.index[0]]:
            for noise_iter in range(1):
                print(hrzn,cntys,noise_iter)
                ndata_df=data_df.copy(deep=True)
                for cnty in data_df.index:
                    ndata_df.loc[cnty,:]=gen_noise(data_df.loc[cnty,:],0,'log')

                dict_df={'cases':ndata_df, 'mob':mobdf}
                cnty_name=cntys#'51059'
                shift_fac=3
                for mul_step in range(1,step_ahead+1):
                    tup_tup_new=gen_tup_mob(dict_df,cnty_name,shift_fac,mul_step)
        #             hrzn_date_str=
                    hrzn_date=datetime.strptime(hrzn,'%Y-%m-%d').date()
                    pred_len=len(tup_tup_new)
                    win=8
                    A=np.zeros([win,pred_len])
                    ii=0
                    for tup in tup_tup_new:
                        tempdf=dict_df[tup[0]]
                        tempdf=tempdf.loc[:,tempdf.columns<=hrzn_date]#+timedelta(weeks=1)
                        try:
                            A[:,ii]=tempdf.loc[tup[1],:].shift(tup[2]).fillna(0).values[-win:]
                        except:
                            continue
                        ii+=1

                    y=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date][-win:]#+timedelta(weeks=1)

                    cols=list(tup_tup_new)
                    idx=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date][-win:].index#+timedelta(weeks=1)
                    Adf=pd.DataFrame(data=A,columns=cols,index=idx)

                    cols=list(tup_tup_new)
                    idx=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date+timedelta(weeks=mul_step)][-win:].index#+timedelta(weeks=1)
                    yf=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date+timedelta(weeks=mul_step)][-win:]#+timedelta(weeks=1)

                    # train model
                    target=y
                    data=Adf
                    try:
                        model, best_features=forward_sel_solver(data,target)
                    except:
                        continue
                        #pdb.set_trace()
                    y_pred=model.predict()
                    y_pred=np.exp(y_pred)
                    y_true=target.values
                    y_true=np.exp(y_true)
                    train_err_mape=get_mape(y_pred,y_true)
                    train_err_mse=get_mse(y_pred,y_true)
                    # forecast
                    nc_idx=model.params.index[1:]
                    coeffs=model.params.values+0.1*np.random.randn(1000,len(model.params))
                    yfh=np.matmul(coeffs[:,1:],Adf.loc[hrzn_date,model.params.index[1:]].values)+coeffs[:,0]
                    fct_week=hrzn_date+timedelta(weeks=mul_step)
                    # y[hrzn_date+timedelta(weeks=1)]
            #         print('{}-step ahead: data available till the {}'.format(mul_step,hrzn_date))
            #         try:
            #             print('forecast for week {}: yfh={}, y={}'.format(fct_week,yfh,yf[fct_week]))
            #             print('forecast for week {}: casesf={}, cases={}'.format(fct_week,np.exp(yfh),np.exp(yf[fct_week])))
            #             print('MAPE={}'.format(mape_met))
            #         except:
            #             print('forecast for week {}: yfh={}, y={}'.format(fct_week,yfh,np.nan))
            #             print('forecast for week {}: casesf={}, cases={}'.format(fct_week,np.exp(yfh),np.nan))

                    # store result
                    resdf=pd.DataFrame(columns=['method','cnty','horizon','fct_date','step_ahead','iter','fct','true','fct_mape','train_mape','train_mse'],index=None)
                    resdf.loc[0,'method']='mob'
                    resdf.loc[0,'cnty']=cnty_name
                    resdf.loc[0,'horizon']=hrzn_date
                    resdf.loc[0,'fct_date']=fct_week
                    resdf.loc[0,'fct_mean']=np.exp(yfh).mean()
                    resdf.loc[0,'fct_std']=np.exp(yfh).std()
                    resdf.loc[0,'step_ahead']='{}-step_ahead'.format(mul_step)
                    resdf.loc[0,'iter']=noise_iter
                    resdf.loc[0,'train_mape']=train_err_mape
                    resdf.loc[0,'train_mse']=train_err_mse



                    try:
                        resdf.loc[0,'true']=np.exp(yf[fct_week])
                        mape_met=100*np.abs((np.exp(yfh)-np.exp(yf[fct_week]))/np.exp(yf[fct_week]))
                        resdf.loc[0,'fct_mape']=mape_met

                    except:
                        resdf.loc[0,'true']=np.nan
                        resdf.loc[0,'fct_mape']=np.nan
                    cntyresdf=cntyresdf.append(resdf,ignore_index=True)
    
    outdir='../output/mob/'
    if not os.path.exists(outdir):
            os.makedirs(outdir)

    outfile=outdir+cnty_name+'_'+hrzn+'_mob.csv'
    cntyresdf.to_csv(outfile,index=None,mode='w')
    return cntyresdf

def predictor_exog(dict_df,cnty_list,hrzn_date_str,step_ahead):
    cntyresdf=pd.DataFrame(columns=['method','cnty','horizon','fct_date','fct','true'],index=None)
    for hrzn in hrzn_date_str:
        for cntys in cnty_list:#[data_df.index[0]]:
            for noise_iter in range(1):
                print(hrzn,cntys,noise_iter)
                data_df=dict_df['cases'].copy('deep')
                #for cnty in data_df.index:
                ndata_df=gen_noise(data_df,0,'log')
                dict_df['cases']=ndata_df
                #dict_df={'cases':ndata_df, 'sdi':mobdf}
                
                cnty_name=cntys#'51059'
                shift_fac=3
                for mul_step in range(1,step_ahead+1):
                    tup_tup_new=gen_tup(dict_df,cnty_name,shift_fac,mul_step)

        #             hrzn_date_str=
                    hrzn_date=datetime.strptime(hrzn,'%Y-%m-%d').date()
                    pred_len=len(tup_tup_new)
                    win=8
                    A=np.zeros([win,pred_len])
                    ii=0
                    for tup in tup_tup_new:
                        tempdf=dict_df[tup[0]]
                        tempdf=tempdf.loc[:,tempdf.columns<=hrzn_date]#+timedelta(weeks=1)
                        try:
                            A[:,ii]=tempdf.loc[tup[1],:].shift(tup[2]).fillna(0).values[-win:]
                        except:
                            continue
                            #pdb.set_trace()
                        ii+=1

                    y=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date][-win:]#+timedelta(weeks=1)

                    cols=list(tup_tup_new)
                    idx=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date][-win:].index#+timedelta(weeks=1)
                    Adf=pd.DataFrame(data=A,columns=cols,index=idx)

                    cols=list(tup_tup_new)
                    idx=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date+timedelta(weeks=mul_step)][-win:].index#+timedelta(weeks=1)
                    yf=ndata_df.loc[cnty_name,data_df.columns<=hrzn_date+timedelta(weeks=mul_step)][-win:]#+timedelta(weeks=1)

                    # train model
                    target=y
                    data=Adf
                    model, best_features=forward_sel_solver(data,target)
                    y_pred=model.predict()
                    y_pred=np.exp(y_pred)
                    y_true=target.values
                    y_true=np.exp(y_true)
                    train_err_mape=get_mape(y_pred,y_true)
                    train_err_mse=get_mse(y_pred,y_true)
                    # forecast
                    nc_idx=model.params.index[1:]
                    coeffs=model.params.values+0.1*np.random.randn(1000,len(model.params))
                    yfh=np.matmul(coeffs[:,1:],Adf.loc[hrzn_date,model.params.index[1:]].values)+coeffs[:,0]
                    fct_week=hrzn_date+timedelta(weeks=mul_step)
                    # y[hrzn_date+timedelta(weeks=1)]
            #         print('{}-step ahead: data available till the {}'.format(mul_step,hrzn_date))
            #         try:
            #             print('forecast for week {}: yfh={}, y={}'.format(fct_week,yfh,yf[fct_week]))
            #             print('forecast for week {}: casesf={}, cases={}'.format(fct_week,np.exp(yfh),np.exp(yf[fct_week])))
            #             print('MAPE={}'.format(mape_met))
            #         except:
            #             print('forecast for week {}: yfh={}, y={}'.format(fct_week,yfh,np.nan))
            #             print('forecast for week {}: casesf={}, cases={}'.format(fct_week,np.exp(yfh),np.nan))

                    # store result
                    resdf=pd.DataFrame(columns=['method','cnty','horizon','fct_date','step_ahead','iter','fct','true','fct_mape','train_mape','train_mse'],index=None)
                    resdf.loc[0,'method']='spatial_mob'
                    resdf.loc[0,'cnty']=cnty_name
                    resdf.loc[0,'horizon']=hrzn_date
                    resdf.loc[0,'fct_date']=fct_week
                    resdf.loc[0,'fct_mean']=np.exp(yfh).mean()
                    resdf.loc[0,'fct_std']=np.exp(yfh).std()
                    resdf.loc[0,'step_ahead']='{}-step_ahead'.format(mul_step)
                    resdf.loc[0,'iter']=noise_iter
                    resdf.loc[0,'train_mape']=train_err_mape
                    resdf.loc[0,'train_mse']=train_err_mse

                    try:
                        resdf.loc[0,'true']=np.exp(yf[fct_week])
                        mape_met=100*np.abs((np.exp(yfh)-np.exp(yf[fct_week]))/np.exp(yf[fct_week]))
                        resdf.loc[0,'fct_mape']=mape_met

                    except:
                        resdf.loc[0,'true']=np.nan
                        resdf.loc[0,'fct_mape']=np.nan
                    cntyresdf=cntyresdf.append(resdf,ignore_index=True)
    
    outdir='../output/AR_exog/'
    if not os.path.exists(outdir):
            os.makedirs(outdir)
    gtdf=cntyresdf.groupby(['cnty','horizon'])
    for group_name, df_group in gtdf:
        cnty_name=group_name[0]
        hrzn=group_name[1].strftime('%Y-%m-%d')
        outfile=outdir+cnty_name+'_'+hrzn+'_exog.csv'
        df_group.to_csv(outfile,index=None,mode='w')
    return cntyresdf


