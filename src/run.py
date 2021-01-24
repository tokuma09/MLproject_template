import argparse
import datetime
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
sys.path.append('../utils')
from create_logger import create_logger
from data_loader import load_datasets, load_target

from logging_mlflow import logging_result

if __name__ == '__main__':

    # JSTとUTCの差分
    DIFF_JST_FROM_UTC = 9
    now = datetime.datetime.utcnow() + datetime.timedelta(
        hours=DIFF_JST_FROM_UTC)

    # parse command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lasso')
    parser.add_argument('--experiment', type=str,
                        default='Base')  # Base or Tuning
    parser.add_argument('--config', default='../config/default.json')
    options = parser.parse_args()
    config = json.load(open(options.config))

    # create logger
    logger = create_logger(options.model, now)

    # ---------------------
    # load model settings
    # FIXME: use hydra
    # ---------------------

    # model parameters
    if options.model == 'logistic':
        from logistic import train_and_predict
        params = config['logistic_params']

    elif options.model == 'rf':
        from rf_reg import train_and_predict
        params = config['rf_params']

    elif options.model == 'lgbm':
        from lgbm_reg import train_and_predict
        params = config['lgbm_params']

    elif options.model == 'lasso':
        from lasso import train_and_predict
        params = config['lasso_params']

    # ---------------------------------
    # load data
    # ---------------------------------
    feats = config['features']
    target_name = config['target_name']

    X_train_all, X_test = load_datasets(feats)
    y_train_all = load_target(target_name)

    y_preds = []
    scores = []
    models = []

    # -------------------------------
    # train model for each fold
    # -------------------------------

    NUM_FOLDS = 3
    kf = KFold(n_splits=NUM_FOLDS, random_state=0)
    logger.info(f'Number of Folds: {NUM_FOLDS}')

    for ind, (train_index, valid_index) in enumerate(kf.split(X=X_train_all)):

        X_train, X_valid = (X_train_all.iloc[train_index, :],
                            X_train_all.iloc[valid_index, :])
        y_train, y_valid = y_train_all[train_index], y_train_all[valid_index]

        # log
        y_train, y_valid = np.log(y_train), np.log(y_valid)

        y_pred, score, model = train_and_predict(X_train, X_valid, y_train,
                                                 y_valid, X_test, params,
                                                 logger)

        # exp
        y_pred = np.exp(y_pred)
        # save result
        y_preds.append(y_pred)
        models.append(model)
        scores.append(score)

    # -------------------------
    #  CV score
    # -------------------------
    score = np.mean(scores)

    logger.info(f'CV scores: {scores}')
    logger.info(f'CV averaged: {score}')

    # ------------------------------------
    # inspection and logging
    # ------------------------------------
    # TODO: add model inspection functions
    # model_inspect(model, features)

    # save prediction results
    logging_result(options, params, scores, models, logger)

    # -------------------------------------------------------
    # prepare submission data
    # -------------------------------------------------------

    # create submit data
    ID_name = config['ID_name']
    sub = pd.DataFrame(pd.read_csv('../data/input/test.csv')[ID_name])

    y_sub = sum(y_preds) / len(y_preds)
    y_sub = y_sub.reshape(-1, 1)
    print(y_sub.shape)
    if y_sub.shape[1] > 1:
        y_sub = np.argmax(y_sub, axis=1)

    sub[target_name] = y_sub

    sub.to_csv('../data/output/{0}_{1:%Y%m%d%H%M%S}_{2}.csv'.format(
        options.model, now, score),
               index=False,
               header=None)
