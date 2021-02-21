import argparse
import inspect
import os
import re
import time
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

import pandas as pd
import yaml

from GCSOperator import GCSOperator
from global_vars import credential, project_id


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force',
                        '-f',
                        action='store_true',
                        help='Overwrite existing files')

    parser.add_argument('--cloud',
                        '-c',
                        action='store_true',
                        help='save features to GCS')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) \
                and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite, cloud):

    if cloud:

        # get bucket name
        f = open("../config/config.yaml", "r+")
        config = yaml.safe_load(f)
        bucket_name = config['bucket_name']
        gcso = GCSOperator(project_id, credential, bucket_name)

        # ファイルの有無を確認する。
        for f in get_features(namespace):
            train_path = os.path.join(
                f.train_path.split('/')[-2],
                f.train_path.split('/')[-1])

            test_path = os.path.join(
                f.test_path.split('/')[-2],
                f.test_path.split('/')[-1])

            if gcso.is_exist(train_path) and gcso.is_exist(
                    test_path) and not overwrite:

                print(f.name, 'was skipped')
            else:
                f.run().save()
    else:

        for f in get_features(namespace):
            if os.path.exists(f.train_path) and os.path.exists(
                    f.test_path) and not overwrite:
                print(f.name, 'was skipped')
            else:
                f.run().save()


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'

    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub("([A-Z])", lambda x: "_" + x.group(1).lower(),
                               self.__class__.__name__).lstrip('_')

        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = os.path.join(self.dir, f'{self.name}_train.feather')
        self.test_path = os.path.join(self.dir, f'{self.name}_test.feather')

    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))

    def load(self):
        self.train = pd.read_feather(str(self.train_path))
        self.test = pd.read_feather(str(self.test_path))
