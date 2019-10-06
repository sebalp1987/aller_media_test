import pandas as pd

from prediction.pred_model import PredictionModel
from preprocessing import etl
from resources import STRING

etl.EtlJob(train_file=True, ar_lags=5, ar_min_lag=1, ma_ss_lag=[60], variable_analysis=False).run()
etl.EtlJob(train_file=False, ar_lags=5, ar_min_lag=1, ma_ss_lag=[60]).run()
'''
df = pd.read_csv(STRING.test_model, sep=';')
df = PredictionModel(df=df)
df = df.preprocessing_test(df)
pred = df.prediction(df)
pred = df.post_process(pred)
'''