import os
import pandas as pd
import yaml
from GCSOperator import GCSOperator
from global_vars import project_id


def load_datasets(feats, base_dir, cloud=False):
    if cloud:
        # get bucket name
        f = open(os.path.join(base_dir, "config/config.yaml"), "r+")
        config = yaml.safe_load(f)
        bucket_name = config['bucket_name']
        gcso = GCSOperator(project_id, bucket_name)

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

    else:

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


def load_target(target_name, base_dir, cloud=False):
    if cloud:
        # get bucket name
        f = open(os.path.join(base_dir, "config/config.yaml"), "r+")
        config = yaml.safe_load(f)
        bucket_name = config['bucket_name']
        gcso = GCSOperator(project_id, bucket_name)

        train = pd.read_feather(gcso.get_fullpath('data/input/train.feather'))
        y_train = train[target_name]

    else:
        input_dir = os.path.join(base_dir, 'data/input')
        train = pd.read_feather(os.path.join(input_dir, 'train.feather'))
        y_train = train[target_name]

    return y_train
