import pandas as pd
import numpy as np

from .feature_base import Feature, get_arguments, generate_features

Feature.dir = '../features'  # set dir
train = pd.read_feather('../data/input/train.feather')
test = pd.read_feather('../data/input/test.feather')


class Pclass(Feature):
    def create_features(self):
        self.train['Pclass'] = train['Pclass']
        self.test['Pclass'] = test['Pclass']


class Sex(Feature):
    def __init__(self):
        super().__init__()
        self.mapping = {"male": 0, "female": 1}

    def create_features(self):
        self.train['Sex'] = train['Sex'].map(self.mapping)
        self.test['Sex'] = test['Sex'].map(self.mapping)


class Age(Feature):
    def create_features(self):
        data = train.append(test)
        age_mean = data['Age'].mean()
        age_std = data['Age'].std()
        self.train['Age'] = pd.qcut(train['Age'].fillna(
            np.random.randint(age_mean - age_std, age_mean + age_std)),
                                    5,
                                    labels=False)
        self.test['Age'] = pd.qcut(test['Age'].fillna(
            np.random.randint(age_mean - age_std, age_mean + age_std)),
                                   5,
                                   labels=False)


class FamilySize(Feature):
    def create_features(self):
        self.train['FamilySize'] = train['Parch'] + train['SibSp'] + 1
        self.test['FamilySize'] = test['Parch'] + test['SibSp'] + 1


class Embarked(Feature):
    def create_features(self):
        # 最頻値で欠損を埋める
        self.train['Embarked'] = train['Embarked'].fillna(('S')).map({
            'S': 0,
            'C': 1,
            'Q': 2
        }).astype(int)
        self.test['Embarked'] = test['Embarked'].fillna(('S')).map({
            'S': 0,
            'C': 1,
            'Q': 2
        }).astype(int)


class Fare(Feature):
    def create_features(self):
        # 欠損をmedianで埋める
        # 分位点を整数で与える
        data = train.append(test)
        fare_median = data['Fare'].median()
        self.train['Fare'] = pd.qcut(train['Fare'].fillna(fare_median),
                                     4,
                                     labels=False)
        self.test['Fare'] = pd.qcut(test['Fare'].fillna(fare_median),
                                    4,
                                    labels=False)


class IsCabin(Feature):
    def create_features(self):
        train['IsCabin'] = 1
        train.loc[train['Cabin'].isnull(), 'IsCabin'] = 0
        test['IsCabin'] = 1
        test.loc[test['Cabin'].isnull(), 'IsCabin'] = 0
        self.train['IsCabin'] = train['IsCabin']
        self.test['IsCabin'] = test['IsCabin']


if __name__ == '__main__':
    args = get_arguments()

    # generate features
    generate_features(globals(), args.force)
