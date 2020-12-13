from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import mlflow


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, params,
                      logger):

    # データセットを生成する
    model = Lasso(**params)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_valid)
    score = np.sqrt(mean_squared_error(y_valid, y_val_pred))

    # テストデータを予測する
    y_pred = model.predict(X_test)

    # logging
    with mlflow.start_run():

        # logging run_id
        run_id = mlflow.active_run().info.run_id
        logger.info(f'Model Dir: {run_id}')

        mlflow.log_params(params)
        mlflow.log_metric("RMSE", score)
        mlflow.sklearn.log_model(model, "model")
        # input_exampleで利用した変数名とサンプルデータがわかるようになる。
        mlflow.sklearn.log_model(model,
                                 "model",
                                 input_example=X_train.head(1).to_dict())

    return y_pred, score, model
