from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

# from sklearn.mixture import GaussianMixture

path = "../../input/tabular-playground-series-jan-2021/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
submission = pd.read_csv(path + "sample_submission.csv")

X = train.drop(["id", "target"], axis=1)
y = train["target"]

X_test = test.drop(["id"], axis=1)


def normalize_transformer(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    quantile = QuantileTransformer(output_distribution="normal")
    normal_train = quantile.fit_transform(train)
    normal_test = quantile.fit_transform(test)
    normal_train = pd.DataFrame(normal_train, columns=train.columns)
    normal_test = pd.DataFrame(normal_test, columns=test.columns)
    return normal_train, normal_test


def tf(train: pd.DataFrame):
    X_cp = train.copy()
    X_tf = X_cp.apply(lambda x: np.e ** (x))
    X_tf.columns = [x + "_tf" for x in X_cp.columns]
    X_tf_final = pd.concat([X_cp, X_tf], axis=1)
    X_tf_final = X_tf_final.transform(lambda x: 1 / (1 + np.e ** (-x)))
    return X_tf_final.drop(
        columns=[col for col in X_tf_final.columns if "_tf" not in col]
    )


X, X_test = normalize_transformer(X, X_test)
