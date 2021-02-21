import importlib
import os
import pickle
import sys

import hydra
import neptune
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.append('../utils')

from data_loader import load_datasets, load_target
from logging_metrics import logging_classification

# global variable

NUM_FOLDS = 3
API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
scoring = accuracy_score  # evaluation metrics


@hydra.main(config_path='../config', config_name='config')
def run(config: DictConfig) -> None:

    # -------------------------
    #  Settings
    # -------------------------

    # get base directory
    base_dir = os.path.dirname(hydra.utils.get_original_cwd())

    # load training API
    module = importlib.import_module(config['model']['file'])

    # ---------------------------------
    # load data
    # ---------------------------------

    feats = config['features']
    target_name = config['target_name']
    params = dict(config['model']['parameters'])

    X_train_all, X_test = load_datasets(feats, base_dir=base_dir)
    y_train_all = load_target(target_name, base_dir=base_dir)

    # start logging
    neptune.init(api_token=API_TOKEN,
                 project_qualified_name='tokuma09/Example')
    neptune.create_experiment(params=params,
                              name='sklearn-quick',
                              tags=[config['model']['name']])

    # ---------------------------
    # train model using CV
    # ---------------------------
    print('***** Train model *****')

    y_test_preds = []
    oof_preds = []
    scores = []
    models = []

    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)

    for ind, (train_index,
              valid_index) in enumerate(kf.split(X=X_train_all,
                                                 y=y_train_all)):

        X_train, X_valid = (X_train_all.iloc[train_index, :],
                            X_train_all.iloc[valid_index, :])
        y_train, y_valid = y_train_all[train_index], y_train_all[valid_index]

        res = module.train_and_predict(X_train, X_valid, y_train, y_valid,
                                       X_test, params, ind, scoring)

        # for evaluation and stacking
        if res['y_val_pred'].ndim > 1:
            y_val_pred = np.argmax(res['y_val_pred'], axis=1)
        else:
            y_val_pred = res['y_val_pred']

        oof_pred = pd.DataFrame([y_valid.index, y_val_pred]).T
        oof_pred.columns = ['index', 'pred']

        # save result
        y_test_preds.append(res['y_test_pred'])
        oof_preds.append(oof_pred)
        models.append(res['model'])
        scores.append(res['score'])

        # logging result
        logging_classification(y_valid, res['y_val_pred'])

    # -------------------------
    #  CV score
    # -------------------------
    print('***** log CV score *****')
    score = np.mean(scores)

    neptune.log_metric('CV score', score)
    for i in range(NUM_FOLDS):
        neptune.log_metric('fold score', scores[i])

    # ---------------------------------
    # save model
    # ---------------------------------
    print('***** Save models *****')
    for ind, model in enumerate(models):
        fname = f'{neptune.get_experiment().id}_model_{ind}.pkl'
        fpath = os.path.join(base_dir, 'models', fname)

        with open(fpath, mode='wb') as fp:
            pickle.dump(model, fp)

    # ---------------------------------
    # save oof result
    # ---------------------------------
    print('***** Save oof *****')
    # concat oof result and save
    df_oof = pd.concat(oof_preds)
    df_oof = df_oof.sort_values(by='index').reset_index(drop=True)

    # for technical reason
    df_temp = pd.DataFrame(df_oof['pred'])
    df_temp.columns = [f'pred_{neptune.get_experiment().id}']

    df_temp.to_feather(
        os.path.join(
            base_dir,
            f"features/valid_pred_{neptune.get_experiment().id}.feather"))

    # ---------------------------------
    # prepare submission
    # ---------------------------------
    print('***** Prepare submission *****')
    # aggregate result
    y_sub = sum(y_test_preds) / len(y_test_preds)
    if y_sub.shape[1] > 1:
        y_sub = np.argmax(y_sub, axis=1)

    # prepare submit data
    ID_name = config['ID_name']
    sub = pd.DataFrame(
        pd.read_csv(os.path.join(base_dir, 'data/input/test.csv'))[ID_name])

    sub[target_name] = y_sub
    sub.to_csv(os.path.join(
        base_dir, f'data/output/test_pred_{neptune.get_experiment().id}.csv'),
               index=False,
               header=None)

    neptune.stop()


if __name__ == '__main__':
    run()
