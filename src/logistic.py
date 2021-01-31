from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, params,
                      logger):

    # データセットを生成する
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_valid)
    score = accuracy_score(y_valid, y_val_pred)

    # テストデータを予測する
    y_pred = model.predict_proba(X_test)

    return y_pred, score, model
