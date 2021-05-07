import os
import pandas as pd
import yaml
from GCSOperator import GCSOperator


def load_local_datasets(feats, base_dir):
    """load features from local

    Parameters
    ----------
    feats : list
        contains feature names
    base_dir : str
        project base directory

    Returns
    -------
    X_train: pd.DataFrame
        training features
    X_test: pd.DataFrame
        test features
    """

    # load data
    feature_dir = os.path.join(base_dir, 'features')

    dfs = [
        pd.read_feather(os.path.join(feature_dir, f'{f}_train.feather'))
        for f in feats
    ]
    X_train = pd.concat(dfs, axis=1, sort=False)

    dfs = [
        pd.read_feather(os.path.join(feature_dir, f'{f}_test.feather'))
        for f in feats
    ]
    X_test = pd.concat(dfs, axis=1, sort=False)

    return X_train, X_test


def load_cloud_datasets(feats, base_dir):
    """load features from cloud

    Parameters
    ----------
    feats : list
        contains feature names
    base_dir : str
        project base directory

    Returns
    -------
    X_train: pd.DataFrame
        training features
    X_test: pd.DataFrame
        test features
    """

    # setup GCS operator
    f = open(os.path.join(base_dir, "config/config.yaml"), "r+")
    config = yaml.safe_load(f)
    bucket_name = config['bucket_name']
    gcso = GCSOperator(config['project_id'], bucket_name)

    # load data
    dfs = [
        pd.read_feather(gcso.get_fullpath(f'features/{f}_train.feather'))
        for f in feats
    ]
    X_train = pd.concat(dfs, axis=1, sort=False)

    dfs = [
        pd.read_feather(gcso.get_fullpath(f'features/{f}_test.feather'))
        for f in feats
    ]
    X_test = pd.concat(dfs, axis=1, sort=False)

    return X_train, X_test


def load_datasets(feats, base_dir, cloud=False):
    """load features from local or cloud

    Parameters
    ----------
    feats : list
        contains feature names
    base_dir : str
        project base directory
    cloud : bool, optional
        load from GCS, by default False

    Returns
    -------
    X_train: pd.DataFrame
        training features
    X_test: pd.DataFrame
        test features
    """

    if cloud:
        X_train, X_test = load_cloud_datasets(feats, base_dir)
    else:
        X_train, X_test = load_local_datasets(feats, base_dir)

    return X_train, X_test


def load_cloud_target(target_name, base_dir):
    """load_target load target from cloud

    Parameters
    ----------
    target_name : str
        target name
    base_dir : str
        project base directory

    Returns
    -------
    y_train: pd.DataFrame
        target data
    """

    # setup GCS operator
    f = open(os.path.join(base_dir, "config/config.yaml"), "r+")
    config = yaml.safe_load(f)
    bucket_name = config['bucket_name']
    gcso = GCSOperator(config['project_id'], bucket_name)

    # load data
    train = pd.read_feather(gcso.get_fullpath('data/input/train.feather'))
    y_train = train[target_name]

    return y_train


def load_local_target(target_name, base_dir):
    """load_target load target from local

    Parameters
    ----------
    target_name : str
        target name
    base_dir : str
        project base directory

    Returns
    -------
    y_train: pd.DataFrame
        target data
    """

    input_dir = os.path.join(base_dir, 'data/input')

    train = pd.read_feather(os.path.join(input_dir, 'train.feather'))
    y_train = train[target_name]

    return y_train


def load_target(target_name, base_dir, cloud=False):
    """load_target load target from local or cloud

    Parameters
    ----------
    target_name : str
        target name
    base_dir : str
        project base directory
    cloud : bool, optional
        load from GCS, by default False

    Returns
    -------
    y_train: pd.DataFrame
        target data
    """

    if cloud:
        y_train = load_cloud_target(target_name, base_dir)
    else:
        y_train = load_local_target(target_name, base_dir)
    return y_train
