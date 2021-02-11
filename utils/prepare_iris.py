import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

RANDOM_STATE = 1


def main():

    # prepare data
    iris = load_iris()
    df = pd.DataFrame(iris.data)

    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df.columns = col_names

    df['target'] = iris.target

    # split train and test
    df_train, df_test = train_test_split(df,
                                         test_size=0.3,
                                         random_state=RANDOM_STATE,
                                         stratify=df['target'])

    # set index
    df_train = df_train.reset_index(drop=True).reset_index()
    df_test = df_test.reset_index(drop=True).reset_index()
    df_test['index'] = df_test['index'] + len(df_train)

    # save data
    df_train.to_csv('../data/input/train.csv', index=False)
    df_test.to_csv('../data/input/test.csv', index=False)

    return None


if __name__ == '__main__':
    main()
