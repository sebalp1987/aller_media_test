import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from resources import STRING

sns.set()
file_list = [filename for filename in os.listdir(STRING.train_model) if
             filename.endswith('.csv')]

df = pd.read_csv(STRING.train_model + file_list[0], sep=';', encoding='utf-8')

y = df['WinningBid'].values

df = df.drop(['key_id', 'WinningBid'], axis=1)
n_cols = df.shape[1]

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), columns=df.columns)

train_x, test_x = df.values[0: int(len(df)*0.70), :], df.values[int(len(df)*0.70):, :]
train_y, test_y = y[0: int(len(df)*0.70)], y[int(len(df)*0.70):]

train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# LSTM
model = Sequential()
model.add(LSTM(6, input_shape=(train_x.shape[1], train_x.shape[2]), activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(lr=0.001))
print("Inputs: " + str(model.input_shape))
print("Outputs: " + str(model.output_shape))
print("Actual input: " + str(train_x.shape))
print("Actual output:" + str(train_y.shape))
history = model.fit(train_x, train_y, epochs=10, validation_data=(test_x, test_y), verbose=True,
                    shuffle=False, steps_per_epoch=train_x.shape[0] // 10000,
                    validation_steps=test_x.shape[0] // 10000)

# PLOT
plot.plot(history.history['loss'], label='train')
plot.plot(history.history['val_loss'], label='test')
plot.legend()
plot.show()


# PREDICTIONS
yhat = model.predict(test_x)
test_y = test_y.reshape(len(test_y), 1)
rmse = np.sqrt(mean_squared_error(test_y, yhat))
print('RMSE % .2f' % rmse)
