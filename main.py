import pandas as pd

from prediction.pred_model import PredictionModel
from preprocessing import etl
from resources import STRING

etl.EtlJob(train_file=True, ar_lags=5, ar_min_lag=1, ma_ss_lag=[60], variable_analysis=False).run()
etl.EtlJob(train_file=False, ar_lags=5, ar_min_lag=1, ma_ss_lag=[60]).run()

df = pd.read_csv(STRING.test_model, sep=';')
id_index = df[['UserID']]
df = df.drop(['key_id', 'UserID'], axis=1)

pred_model = PredictionModel(df=df)
df_pre = pred_model.preprocessing_test()
pred = df_pre.prediction()
pred_model.post_process(pred, id_index, name='gb')
