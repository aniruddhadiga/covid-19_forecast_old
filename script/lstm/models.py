from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout, Add, TimeDistributed
from tensorflow.keras import regularizers

def two_branch_lstm(ili_input_shape, ght_input_shape, hidden_rnn, hidden_dense, output_dim):
    inp_ili = Input(shape=ili_input_shape)
    ili = Sequential()
    ili = LSTM(input_shape=ili_input_shape,units=hidden_rnn,return_sequences=True)(inp_ili)
    ili = LSTM(input_shape=ili_input_shape,units=hidden_rnn,return_sequences=False)(ili)
    ili = Dense(units=hidden_dense,activation='linear')(ili)
    
    inp_ght = Input(shape=ght_input_shape)
    ght = Sequential()
    ght = LSTM(input_shape=ght_input_shape,units=hidden_rnn,return_sequences=True)(inp_ght)
    ght = LSTM(input_shape=ght_input_shape,units=hidden_rnn,return_sequences=False)(ght)
    ght = Dense(units=hidden_dense,activation='linear')(ght)
    
    merge = Add()([ili, ght])
    merge = Dropout(0.2)(merge, training=True)
    out = Dense(units=output_dim,activation='linear',kernel_regularizer=regularizers.l2(0.01))(merge)
    return Model(inputs=[inp_ili,inp_ght], outputs=out, name='two_branch_lstm')

def lstm_mcdropout(input_shape, hidden_rnn, hidden_dense, output_dim, activation):
    inp = Input(shape=input_shape)
    left = Sequential()
    left = LSTM(input_shape=input_shape,units=hidden_rnn,return_sequences=False)(inp)
    left = Dense(units=hidden_dense,activation=activation)(left)
    left = Dropout(0.2)(left, training=True)   
    out = Dense(units=output_dim,activation='linear',kernel_regularizer=regularizers.l2(0.01))(left)
    return Model(inputs=inp, outputs=out, name='lstm_mcdropout') 
