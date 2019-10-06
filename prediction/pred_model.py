from sklearn.externals import joblib
import os
from resources import STRING
import pandas as pd


class PredictionModel:
    def __init__(self, df):
        self._dict_models = joblib.load(
            os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                         "aller_media", "models", "model.pkl"))
        self._model = self._dict_models.get("model")
        self._scale_param = self._dict_models.get("param_scale")
        self.df = df

    def preprocessing_test(self):
        print(self.df.columns.values.tolist())
        # Scale
        df = pd.DataFrame(self._scale_param.transform(self.df), columns=self.df.columns)
        return df

    @staticmethod
    def prediction(df):
        pred = model.predict(df)
        pred = pd.DataFrame(pred, columns=['y_pred'])
        return pred

    @staticmethod
    def post_process(pred):
        return pred

