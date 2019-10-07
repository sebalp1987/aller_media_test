import pandas as pd
import os

from prediction.pred_model import PredictionModel
from preprocessing import etl
from resources import STRING


etl.EtlJob(train_file=True, ar_lags=5, ar_min_lag=1, ma_ss_lag=[60], variable_analysis=False).run()
etl.EtlJob(train_file=False, ar_lags=5, ar_min_lag=1, ma_ss_lag=[60]).run()


file_list = [filename for filename in os.listdir(STRING.test_model) if
             filename.endswith('.csv')]

df = pd.read_csv(STRING.test_model + file_list[0], sep=';')
id_index = df[['UserID']]
df = df.drop(['key_id', 'UserID'], axis=1)

pred_model = PredictionModel(df=df)
pred_model.preprocessing_test()
pred = pred_model.prediction()
pred_model.post_process(pred, id_index, name='lgbm')
