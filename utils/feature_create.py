import pandas as pd
import numpy as np

from feature_base import Feature, get_arguments, generate_features

Feature.dir = '../features'  # set dir
train = pd.read_feather('../data/input/train.feather')
test = pd.read_feather('../data/input/test.feather')
all = pd.concat([train, test])
num_train = len(train)


class Sex(Feature):
    def __init__(self):
        super().__init__()
        self.mapping = {"male": 0, "female": 1}

    def create_features(self):
        col_name = 'Sex'
        data = all[col_name].map(self.mapping)
        self.train[col_name] = data.iloc[:num_train]
        self.test[col_name] = data.iloc[num_train:]


if __name__ == '__main__':
    args = get_arguments()

    # generate features
    generate_features(globals(), args.force)
