import os
import pandas as pd


def load_datasets(feats, base_dir):
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


def load_target(target_name, base_dir):
    input_dir = os.path.join(base_dir, 'data/input')
    train = pd.read_feather(os.path.join(input_dir, 'train.feather'))
    y_train = train[target_name]
    return y_train
