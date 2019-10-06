from sklearn.externals import joblib
import os
import pandas as pd
from resources import STRING

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
        df = pd.DataFrame(self._scale_param.transform(self), columns=self.columns)
        return df

    def prediction(self):
        pred = self._model.predict(self)
        pred = pd.DataFrame(pred, columns=['y_pred'])
        return pred

    @staticmethod
    def post_process(pred, key_df, name=''):
        key_df = pd.concat([key_df, pred], axis=1)
        if not name:
            key_df.to_csv(STRING.submission, sep=';', index=False)
        else:
            key_df.to_csv(STRING.model_output_path + 'model_' + name + '.csv', sep=';', index=False)
        return 0

