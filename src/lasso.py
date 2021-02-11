from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import numpy as np


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, params):

    # データセットを生成する
    model = Lasso(**params)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_valid)
    score = np.sqrt(mean_squared_error(y_valid, y_val_pred))

    # テストデータを予測する
    y_pred = model.predict(X_test)

    return y_pred, score, model
