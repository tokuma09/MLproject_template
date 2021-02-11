import pandas as pd
import numpy as np

from feature_base import Feature, get_arguments, generate_features

Feature.dir = '../features'  # set dir
train = pd.read_feather('../data/input/train.feather')
test = pd.read_feather('../data/input/test.feather')
all = pd.concat([train, test])
num_train = len(train)


class Sepal_length(Feature):
    def __init__(self):
        super().__init__()

    def create_features(self):
        col_name = 'sepal_length'
        self.train[col_name] = all[col_name].iloc[:num_train]
        self.test[col_name] = all[col_name].iloc[num_train:]


class Sepal_width(Feature):
    def __init__(self):
        super().__init__()

    def create_features(self):
        col_name = 'sepal_width'
        self.train[col_name] = all[col_name].iloc[:num_train]
        self.test[col_name] = all[col_name].iloc[num_train:]


class Petal_length(Feature):
    def __init__(self):
        super().__init__()

    def create_features(self):
        col_name = 'petal_length'
        self.train[col_name] = all[col_name].iloc[:num_train]
        self.test[col_name] = all[col_name].iloc[num_train:]


class Petal_width(Feature):
    def __init__(self):
        super().__init__()

    def create_features(self):
        col_name = 'petal_width'
        self.train[col_name] = all[col_name].iloc[:num_train]
        self.test[col_name] = all[col_name].iloc[num_train:]


if __name__ == '__main__':
    args = get_arguments()

    # generate features
    generate_features(globals(), args.force)
