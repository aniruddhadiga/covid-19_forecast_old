import numpy as np
import pandas as pd

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     n_vars = 1 if type(data) is list or np.ndarray else data.shape[1]
#     df = pd.DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
#     # put it all together
#     agg = pd.concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg

def predict_n_point(model, X, steps, scaler, len_scal, nth_scal, n_back, n_feature=1):
    ### N step ahead forecasting (point estimate) ###
    points = []
    p = X
    for i in range(steps):
        pred = model.predict(p.reshape(-1,n_back,n_feature))
        tran_pd = np.asarray([pred[0]]*len_scal).reshape(1,len_scal)
        point = scaler.inverse_transform(tran_pd)
        points.append(point[0][nth_scal])
        p = np.append(p,pred[0])[1:]
    return points

def predict_n_point_by_one_step(model, X, scaler, len_scal, nth_scal, n_back, n_feature=1):
    ### N step ahead forecasting (point estimate) ###
    points = []
    pred = model.predict(X)
    for i in range(pred.shape[1]):
        tran_pd = np.asarray([pred[0][i]]*len_scal).reshape(1,len_scal)
        point = scaler.inverse_transform(tran_pd)
        points.append(point[0][nth_scal])
    return points

# def predict_n_point_mf(model, X, steps, scaler, len_scal, nth_scal, n_back, n_feature=1):
#     ### N step ahead forecasting (point estimate) ###
#     points = []
#     p = X
#     for i in range(steps):
#         pred = model.predict(p.reshape(-1,n_back,n_feature))
#         tran_pd = np.asarray([pred[0]]*len_scal*n_feature).reshape(1,len_scal*n_feature)
#         point = scaler.inverse_transform(tran_pd)
#         points.append(point[0][nth_scal])
#         p = np.append(p,pred[0])[1:]
#     return points

def predict_n_point_mf(model, X, scaler, len_scal, nth_scal, n_back, n_feature=1):
    ### N step ahead forecasting (point estimate) ###
    pred = model.predict(X.reshape(-1,n_back,n_feature))
    tran_pd = np.asarray([pred[0]]*len_scal*n_feature).reshape(1,len_scal*n_feature)
    point = scaler.inverse_transform(tran_pd)
    return point[0][nth_scal]

def predict_n_point_ens(model, X, scaler, len_scal, nth_scal, n_back, n_feature=1):
    ### N step ahead forecasting (point estimate) ###
    pred = model.predict(X.reshape(-1,n_back,n_feature))
    tran_pd = np.asarray([pred[0]]*len_scal*(n_feature+1)).reshape(1,len_scal*(n_feature+1))
    point = scaler.inverse_transform(tran_pd)
    return point[-1][nth_scal]

