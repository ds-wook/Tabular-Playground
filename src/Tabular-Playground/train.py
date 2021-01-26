import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

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

    lgb_params = pd.read_pickle(args.path + "fea_lgb_optuna2.pkl")
    lgb_model = LGBMRegressor(**lgb_params)
    lgb_preds = kfold_model(lgb_model, args.fold, X, y, X_test)

    # xgb_params = pd.read_pickle(args.path + "xgb_optim.pkl")
    # xgb_model = XGBRegressor(**xgb_params)
    # xgb_preds = kfold_model(xgb_model, args.fold, X, y, X_test)

    # cat_params = pd.read_pickle(args.path + "xgb_optim.pkl")
    # cat_model = CatBoostRegressor(**cat_params)
    # cat_preds = kfold_model(cat_model, args.fold, X, y, X_test)

    submission["target"] = lgb_preds
    submission.to_csv(args.path + args.file, index=False)
