import os
import pickle

import mlflow
import numpy as np


def logging_result(options, params, scores, models, logger):
    """logging_result logging model result using MLflow

    Only one type of cv score is allowed.
    Parameters
    ----------
    options : parser
        store model name
    params : dict
        model parameters
    scores : list
        store cv results. only one type of score is allowed.
    models : list of trained models
        store model objects
    logger : logger
    """

    # set experiment directories
    mlflow.set_tracking_uri('../mlruns')
    mlflow.set_experiment(options.experiment)
    logger.info(f'Experiment name: {options.experiment}')

    # temporary save models
    model_paths = [f'../models/model_{i}.pkl' for i in range(len(models))]
    for i, model in enumerate(models):
        with open(model_paths[i], 'wb') as f:
            pickle.dump(model, f)

    with mlflow.start_run():

        run_id = mlflow.active_run().info.run_id
        logger.info(f'Model Dir: {run_id}')

        # logging model name and parameters
        mlflow.log_param({'model': options.model})
        mlflow.log_params(params)

        # logging scores
        for i, score in enumerate(scores):
            mlflow.log_metric(f'fold {i}', score)

        score = np.mean(scores)
        mlflow.log_metric('RMSE', score)

        # save model files
        for item in model_paths:
            mlflow.log_artifact(item)

        # TODO: save inspection results
        # 1. feature importances
        # 2. PDP
        # 3. SHAP

    # remove files
    for item in model_paths:
        os.remove(item)
        # TODO: remove inspection results
