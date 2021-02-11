import lightgbm as lgb
from sklearn.metrics import accuracy_score
import numpy as np
from neptunecontrib.monitoring.lightgbm import neptune_monitor


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, params,
                      fold_ind):

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=['train', 'valid'],
        # 最大で 1000 ラウンドまで学習する
        num_boost_round=1000,
        # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
        early_stopping_rounds=10,
        callbacks=[neptune_monitor(prefix=f'fold_{fold_ind}_')])

    # predict validation data
    y_val_pred = model.predict(X_valid, num_iteration=model.best_iteration)

    score = accuracy_score(y_valid, np.argmax(y_val_pred, axis=1))

    # テストデータを予測する
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    return y_pred, score, model
