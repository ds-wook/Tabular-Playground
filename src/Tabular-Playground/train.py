import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from model.tree_model import kfold_model
from data.datasets import X, X_test, y, submission

np.seterr(divide="ignore", invalid="ignore")

if __name__ == "__main__":
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--path", type=str, help="Input data save path", default="../../res/"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
    parse.add_argument("--fold", type=int, help="Input num_fold", default=5)
    args = parse.parse_args()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=0
    )
    lgb_params = {
        "seed": 2021,
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "feature_pre_filter": False,
        "lambda_l1": 6.540486456085813,
        "lambda_l2": 0.01548480538099245,
        "num_leaves": 256,
        "feature_fraction": 0.52,
        "bagging_fraction": 0.6161835249194311,
        "bagging_freq": 7,
        "min_child_samples": 20,
    }
    lgb_params["learning_rate"] = 0.001
    lgb_params["num_iterations"] = 20000
    lgb_model = LGBMRegressor(**lgb_params)
    lgb_preds = kfold_model(lgb_model, args.fold, X, y, X_test)

    # xgb_params = pd.read_pickle(args.path + "xgb_optuna1.pkl")
    # xgb_params["n_estimators"] = 4000
    # xgb_model = XGBRegressor(**xgb_params)
    # xgb_preds = kfold_model(xgb_model, args.fold, X, y, X_test)

    # cat_params = pd.read_pickle(args.path + "cat_bayesian1.pkl")
    # cat_model = CatBoostRegressor(**cat_params)
    # cat_preds = kfold_model(cat_model, args.fold, X, y, X_test)

    submission["target"] = lgb_preds
    submission.to_csv(args.path + args.file, index=False)
