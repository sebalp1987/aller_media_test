import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns

from keras.utils import to_categorical
from keras import layers, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import L1L2

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from resources import STRING

sns.set()
file_list = [filename for filename in os.listdir(STRING.train_model) if
             filename.endswith('.csv')]

df = pd.read_csv(STRING.train_model + file_list[0], sep=';', encoding='utf-8')

y = df['WinningBid'].values

df = df.drop(['key_id', 'WinningBid'], axis=1)
n_cols = df.shape[1]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), columns=df.columns)

input_tensor = Input(shape=(n_cols, ))
x = layers.Dense(6, activation='relu', kernel_regularizer=L1L2(l1=0., l2=0.01))(input_tensor)
x = layers.Dropout(0.2)(x)
output_tensor = layers.Dense(1, kernel_initializer='normal', activation='linear')(x)

model = Model(input_tensor, output_tensor)
print(model.summary())

model.compile(optimizer=Adam(), loss='mean_squared_error')

skf = TimeSeriesSplit(n_splits=5)

y_pred_score = np.empty(shape=[0, ])
y_true = np.empty(shape=[0, ])

predicted_index = np.empty(shape=[0, ])
for train_index, test_index in skf.split(df, y):
    X_train, X_test = df.loc[train_index].values, df.loc[test_index].values
    y_train, y_test = y[train_index], y[test_index]

    history = model.fit(X_train, y_train, epochs=100, steps_per_epoch=X_train.shape[0] // 10000,
                        validation_steps=X_test.shape[0] // 10000,
                        shuffle=False, verbose=True, callbacks=[EarlyStopping(patience=2)], validation_split=None,
                        )
    prediction_i = model.predict(X_test)

    y_pred_score = np.append(y_pred_score, prediction_i[:, 0], axis=0)
    y_true = np.append(y_true, y_test, axis=0)
    predicted_index = np.append(predicted_index, test_index)
    del X_train, X_test, y_train, y_test

    # Error Plot
    fig, ax = plot.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)
    plot.show()

rmse = np.sqrt(mean_squared_error(y_true, y_pred_score))
print('RMSE % .2f' % rmse)
