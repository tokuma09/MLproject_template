import datetime
import importlib
import os
import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold

sys.path.append('../utils')
from create_logger import create_logger
from data_loader import load_datasets, load_target
from logging_mlflow import logging_result

# JSTとUTCの差分
DIFF_JST_FROM_UTC = 9

NUM_FOLDS = 3


@hydra.main(config_path='../config', config_name='config')
def run(config: DictConfig) -> None:

    # -------------------------
    #  Settings
    # -------------------------

    # create logger
    now = datetime.datetime.utcnow() + datetime.timedelta(
        hours=DIFF_JST_FROM_UTC)

    # create logger
    logger = create_logger(config['model']['name'], now)

    # get base directory
    base_dir = os.path.dirname(hydra.utils.get_original_cwd())

    # load training API
    module = importlib.import_module(config['model']['file'])
    print(module.train_and_predict)

    # ---------------------------------
    # load data
    # ---------------------------------

    feats = config['features']
    target_name = config['target_name']
    params = config['model']['parameters']

    X_train_all, X_test = load_datasets(feats, base_dir=base_dir)
    y_train_all = load_target(target_name, base_dir=base_dir)

    # ---------------------------
    # train model using CV
    # ---------------------------
    y_preds = []
    scores = []
    models = []

    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)

    for ind, (train_index,
              valid_index) in enumerate(kf.split(X=X_train_all,
                                                 y=y_train_all)):

        X_train, X_valid = (X_train_all.iloc[train_index, :],
                            X_train_all.iloc[valid_index, :])
        y_train, y_valid = y_train_all[train_index], y_train_all[valid_index]

        y_pred, score, model = module.train_and_predict(
            X_train, X_valid, y_train, y_valid, X_test, params, logger)

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

    # aggregate result
    y_sub = sum(y_preds) / len(y_preds)
    if y_sub.shape[1] > 1:
        y_sub = np.argmax(y_sub, axis=1)

    # prepare submit data
    ID_name = config['ID_name']
    sub = pd.DataFrame(
        pd.read_csv(os.path.join(base_dir, 'data/input/test.csv'))[ID_name])

    sub[target_name] = y_sub
    sub.to_csv(os.path.join(base_dir,
                            'data/output/{0}_{1:%Y%m%d%H%M%S}_{2}.csv').format(
                                config['model']['name'], now, score),
               index=False,
               header=None)


if __name__ == '__main__':
    run()
