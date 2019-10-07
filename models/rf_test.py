import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from resources import STRING

sns.set()
file_list = [filename for filename in os.listdir(STRING.train_model) if
             filename.endswith('.csv')]
df = pd.read_csv(STRING.train_model + file_list[0], sep=';', encoding='utf-8')

y = df['WinningBid'].values
df = df.drop(['key_id', 'WinningBid'], axis=1)
columns = df.columns

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), columns=df.columns)

min_sample_leaf = round(y.shape[0] * 0.0001)
min_sample_split = min_sample_leaf * 10
model = RandomForestRegressor(n_estimators=500, min_samples_leaf=min_sample_leaf,
                              min_samples_split=min_sample_split, random_state=42, max_depth=None, n_jobs=-1,
                              max_features=5)

skf = TimeSeriesSplit(n_splits=5)

y_pred_score = np.empty(shape=[0, ])
y_true = np.empty(shape=[0, ])
predicted_index = np.empty(shape=[0, ])

for train_index, test_index in skf.split(df, y):
    print('iter')
    X_train, X_test = df.loc[train_index].values, df.loc[test_index].values
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    prediction_i = model.predict(X_test)
    y_pred_score = np.append(y_pred_score, prediction_i, axis=0)
    y_true = np.append(y_true, y_test, axis=0)
    predicted_index = np.append(predicted_index, test_index)
    del X_train, X_test, y_train, y_test

print(y_true)
print(y_pred_score)
rmse = np.sqrt(mean_squared_error(y_true, y_pred_score))
print('RMSE % .2f' % rmse)

# Feature Importance
feature_importance = model.feature_importances_
feature_importance = feature_importance / feature_importance.max()
sorted_idx = np.argsort(feature_importance)
bar_position = np.arange(sorted_idx.shape[0]) + 0.5
plot.barh(bar_position, feature_importance[sorted_idx], align='center')
plot.yticks(bar_position, columns[sorted_idx], fontsize=8)
plot.xlabel('Variable Importance')
plot.show()
