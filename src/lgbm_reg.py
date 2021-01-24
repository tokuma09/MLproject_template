import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, params,
                      logger):

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    # 上記のパラメータでモデルを学習する
    model = lgb.train(
        params,
        lgb_train,
        # モデルの評価用データを渡す
        valid_sets=lgb_eval,
        # 最大で 1000 ラウンドまで学習する
        num_boost_round=1000,
        # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
        early_stopping_rounds=10,
    )

    # predict validation data
    y_val_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    score = np.sqrt(mean_squared_error(y_valid, y_val_pred))

    # テストデータを予測する
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    return y_pred, score, model
