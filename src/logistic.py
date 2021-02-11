from sklearn.linear_model import LogisticRegression
import numpy as np


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

    # train model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # evaluate model on the validation dataset
    y_val_pred = model.predict_proba(X_valid)
    score = scoring(y_valid, np.argmax(y_val_pred, axis=1))

    # predict targets on the test dataset
    y_test_pred = model.predict(X_test)

    # store results
    res = {}
    res['model'] = model
    res['score'] = score
    res['y_val_pred'] = y_val_pred
    res['y_test_pred'] = y_test_pred
    return res
