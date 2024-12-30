#!/usr/bin/env python
# coding: utf-8

# In[80]:


# import lib
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Dropout

from keras.models import save_model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import joblib

import pandas as pd
import numpy as np
import pickle


# In[4]:


df = pd.read_csv("data.csv")
df.set_index("date")


# In[5]:


df.info()


# In[6]:


sc = MinMaxScaler(feature_range=(0, 1))


# In[7]:


df_scaled = sc.fit_transform(df[['price', 'warehouse_capacity', 'truck_capacity']])


# In[8]:


df.info


# In[9]:


windows = 14
X_train = []
y_train = []


# In[11]:


for i in range(windows, len(df)):
    X_train.append(df_scaled[i - windows:i])
    y_train.append(df_scaled[i])


# In[13]:


X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train.shape)
print(y_train.shape)


# In[31]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


# In[46]:


def LSTM_model():
    model = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    layer_1_units=64
    model.add(LSTM(units = layer_1_units, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    
    # Adding a second LSTM layer and some Dropout regularisation
    layer_2_units=128
    model.add(LSTM(units = layer_2_units))
    model.add(Dropout(0.2))
    
    # Adding the output layer
    model.add(Dense(units = y_train.shape[1]))
    
    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return model


# In[65]:


model_LSTM = LSTM_model()

early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)


# In[54]:


# Fitting the RNN to the Training set
epoch_no=64
batch_size=44
history = model_LSTM.fit(X_train, y_train, epochs = epoch_no, batch_size = batch_size, callbacks=[early_stopping],
                         validation_data=(X_val, y_val))


# In[55]:


train_loss = history.history['loss']

plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Training Loss', color='blue')
plt.title('Training Loss Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()


# In[61]:


def RNN_model():
    model = Sequential()
    
    # Adding the first RNN layer and some Dropout regularisation
    layer_1_units = 50
    model.add(SimpleRNN(units=layer_1_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    
    # Adding a second RNN layer and some Dropout regularisation
    layer_2_units = 400
    model.add(SimpleRNN(units=layer_2_units))
    model.add(Dropout(0.2))
    
    # Adding the output layer
    model.add(Dense(units=y_train.shape[1]))
    
    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model


# In[62]:


def GRU_model():
    model = Sequential()
    
    # Adding the first GRU layer and some Dropout regularisation
    layer_1_units = 50
    model.add(GRU(units=layer_1_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    
    # Adding a second GRU layer and some Dropout regularisation
    layer_2_units = 400
    model.add(GRU(units=layer_2_units))
    model.add(Dropout(0.2))
    
    # Adding the output layer
    model.add(Dense(units=y_train.shape[1]))
    
    # Compiling the GRU
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model


# In[70]:


model_rnn = RNN_model()
history = model_rnn.fit(X_train, y_train, epochs = epoch_no, batch_size = batch_size, callbacks=[early_stopping],
                         validation_data=(X_val, y_val))


# In[71]:


train_loss = history.history['loss']

plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Training Loss', color='blue')
plt.title('Training Loss Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()


# In[74]:


model_gru = GRU_model()
history = model_gru.fit(X_train, y_train, epochs = epoch_no, batch_size = batch_size, callbacks=[early_stopping],
                         validation_data=(X_val, y_val))


# In[75]:


train_loss = history.history['loss']

plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Training Loss', color='blue')
plt.title('Training Loss Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()


# In[76]:


# save_model
model_LSTM.save('./model_lstm.h5')
model_rnn.save('./model_rnn.h5')
model_gru.save('./model_gru.h5')


# In[83]:


joblib.dump(sc, 'scaler.pkl')


# In[84]:


type(sc)


# In[ ]:




