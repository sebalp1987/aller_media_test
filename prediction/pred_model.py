from sklearn.externals import joblib
import os
import pandas as pd
from resources import STRING
from keras.models import load_model


class PredictionModel:
    def __init__(self, df, nn=None):
        self._dict_models = joblib.load(
            os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                         "aller_media", "models", "model_rf.pkl"))
        self._model = self._dict_models.get("model")
        self._scale_param = self._dict_models.get("param_scale")
        self._columns = self._dict_models.get("columns")
        self.df = df
        if nn is not None:
            self.nn = load_model(STRING.model_path + 'model_nn.h5')
        else:
            self.nn = None

    def preprocessing_test(self):
        self.df = pd.DataFrame(self._scale_param.transform(self.df), columns=self._columns)
        return self.df

    def prediction(self):
        if self.nn is None:
            pred = self._model.predict(self.df)
        else:
            pred = self.nn.predict(self.df)
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
