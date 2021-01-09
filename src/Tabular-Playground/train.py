import argparse

import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from data.datasets import X, X_test, y, submission
from optim.bayesian_optim import (
    lgb_rmse_eval,
    lgb_parameter,
    cat_rmse_eval,
    cat_parameter,
)
from model.tree_model import kfold_model

np.seterr(divide="ignore", invalid="ignore")

if __name__ == "__main__":
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--path", type=str, help="Input data save path", default="../../res"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
    parse.add_argument("--fold", type=int, help="Input num_fold", default=5)
    args = parse.parse_args()

    lgb_params = {
        "learning_rate": (0.0001, 0.05),
        "reg_lambda": (0, 1),
        "reg_alpha": (0, 1),
        "num_leaves": (100, 200),
        "min_child_samples": (20, 50),
    }

    lgb_bo = lgb_parameter(lgb_rmse_eval, lgb_params)

    lgb_params = {
        "n_estimators": 10000,
        "objective": "regression",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
        "learning_rate": max(min(lgb_bo["learning_rate"], 1), 0),
        "reg_lambda": max(min(lgb_bo["reg_lambda"], 1), 0),
        "reg_alpha": max(min(lgb_bo["reg_alpha"], 1), 0),
        "num_leaves": int(round(lgb_bo["num_leaves"])),
        "min_child_samples": int(round(lgb_bo["min_child_samples"])),
    }

    cat_params = {
        "iterations": 100,
        "loss_function": "RMSE",
        "verbose": False,
        "depth": (4, 10),
        "bagging_temperature": (0.1, 10),
        "l2_leaf_reg": (0.1, 10),
        "learning_rate": (0.1, 0.2),
    }

    cat_bo = cat_parameter(cat_rmse_eval, cat_params)

    cat_params = {
        "iterations": 100,
        "loss_function": "RMSE",
        "verbose": False,
        "depth": int(round(cat_bo["depth"])),
        "bagging_temperature": max(min(cat_bo["bagging_temperature"], 1), 0),
        "l2_leaf_reg": max(min(cat_bo["l2_leaf_reg"], 1), 0),
        "learning_rate": max(min(cat_bo["learning_rate"], 1), 0),
    }
    lgb_model = LGBMRegressor(**lgb_params)
    lgb_preds = kfold_model(lgb_model, args.fold, X, y, X_test)

    cat_model = CatBoostRegressor(**cat_params)
    cat_preds = kfold_model(cat_model, args.fold, X, y, X_test)
    submission["target"] = 0.7 * lgb_preds + 0.3 * cat_preds
    submission.to_csv(args.path + args.file, index=False)
