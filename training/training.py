import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor

from resources import STRING
from config import config

file_list = [filename for filename in os.listdir(STRING.train_model) if
             filename.endswith('.csv')]
df = pd.read_csv(STRING.train_model + file_list[0], sep=';', encoding='utf-8')
parameters = config.gb

y = df['WinningBid'].values
df = df.drop(['key_id', 'WinningBid'], axis=1)
columns = df.columns

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), columns=df.columns)

min_sample_leaf = round(y.shape[0] * 0.01)
min_sample_split = min_sample_leaf * 10
model = GradientBoostingRegressor(min_samples_leaf=min_sample_leaf,
                                  min_samples_split=min_sample_split)

model.set_params(**parameters)
print(model)
model.fit(df, y)
dict_model = {'model': model, 'param_scale': scaler}

joblib.dump(dict_model,
            os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                         "aller_media", "models", "model.pkl"))