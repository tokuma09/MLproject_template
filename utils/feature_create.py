import pandas as pd
import numpy as np
import re as re

from feature_base import Feature, get_arguments, generate_features

Feature.dir = 'features' # set dir



class FamilySize(Feature):
    def create_features(self):
        self.train['FamilySize'] = train['Parch'] + train['SibSp'] + 1
        self.test['FamilySize'] = test['Parch'] + test['SibSp'] + 1





if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')
    # generate features
    generate_features(globals(), args.force)
