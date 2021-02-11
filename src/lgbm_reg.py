import lightgbm as lgb
import numpy as np
from neptunecontrib.monitoring.lightgbm import neptune_monitor


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, params,
                      fold_ind, scoring):
    """train_and_predict

    train and evaluate the model and predict targets.
    The interface is same across any ML algorithms.

    Parameters
    ----------
    X_train : array-like
        features in the training data
    X_valid : array-like
        features in the training data
    y_train : array-like
        ground truth target in the training data
    y_valid : array-like
        ground truth target in the validation data
    X_test : array-like
        features in the test data
    params : dict
        model parameters governed by yaml files.
    fold_ind : int
        fold id

    scoring : function
        evaluation metrics function

    Returns
    -------
    [type]
        [description]
    """

    # create dataset
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    # train model
    model = lgb.train(params,
                      lgb_train,
                      valid_sets=[lgb_train, lgb_eval],
                      valid_names=['train', 'valid'],
                      num_boost_round=1000,
                      early_stopping_rounds=10,
                      callbacks=[neptune_monitor(prefix=f'fold_{fold_ind}_')])

    # evaluate model
    y_val_pred = model.predict(X_valid, num_iteration=model.best_iteration)

    score = scoring(y_valid, y_val_pred)

    # predict target on the test dataset
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # store results
    res = {}
    res['model'] = model
    res['score'] = score
    res['y_val_pred'] = y_val_pred
    res['y_test_pred'] = y_test_pred
    return res
