import pickle

import numpy as np

from optim.bayesian_optim import (
    lgb_rmse_eval,
    lgb_parameter,
    cat_rmse_eval,
    cat_parameter,
    xgb_rmse_eval,
    xgb_parameter,
)

np.seterr(divide="ignore", invalid="ignore")

if __name__ == "__main__":
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
        "learning_rate": max(min(lgb_bo["learning_rate"], 1), 0),
        "reg_lambda": max(min(lgb_bo["reg_lambda"], 1), 0),
        "reg_alpha": max(min(lgb_bo["reg_alpha"], 1), 0),
        "num_leaves": int(round(lgb_bo["num_leaves"])),
        "min_child_samples": int(round(lgb_bo["min_child_samples"])),
    }

    xgb_params = {
        "min_child_weight": (3, 20),
        "gamma": (0, 5),
        "subsample": (0.7, 1),
        "colsample_bytree": (0.1, 1),
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.5),
    }
    xgb_bo = xgb_parameter(xgb_rmse_eval, xgb_params)

    xgb_params = {
        "n_estimators": 10000,
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": xgb_bo["learning_rate"],
        "gamma": xgb_bo["gamma"],
        "subsample": xgb_bo["subsample"],
        "colsample_bytree": xgb_bo["colsample_bytree"],
        "max_depth": int(round(xgb_bo["max_depth"])),
        "min_child_weight": int(round(xgb_bo["min_child_weight"])),
    }

    cat_params = {
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
        "bagging_temperature": cat_bo["bagging_temperature"],
        "l2_leaf_reg": max(min(cat_bo["l2_leaf_reg"], 1), 0),
        "learning_rate": cat_bo["learning_rate"],
    }

    print("LGBM Optimization params: ", lgb_bo)
    print("XGB Optimization params: ", xgb_bo)
    print("CAT Optimization params: ", cat_bo)

    bo_optim = [lgb_bo, xgb_bo, cat_bo]
    bo_name = ["lgb_optim.pkl", "xgb_optim.pkl", "cat_optim.pkl"]

    for b, n in zip(bo_optim, bo_name):
        with open("../../res/" + n, "wb") as f:
            pickle.dump(b, f)
