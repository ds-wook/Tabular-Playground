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
        "max_depth": (4, 12),
        "reg_lambda": (1e-13, 5),
        "reg_alpha": (1e-13, 2),
        "colsample_bytree": (0.001, 1),
        "subsample": (0.001, 1),
        "num_leaves": (100, 200),
        "min_child_samples": (10, 50),
    }
    lgb_bo = lgb_parameter(lgb_rmse_eval, lgb_params)

    lgb_params = {
        "n_estimators": 20000,
        "objective": "regression",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.005,
        "max_depth": int(round(lgb_bo["max_depth"])),
        "reg_lambda": max(min(lgb_bo["reg_lambda"], 1), 0),
        "reg_alpha": max(min(lgb_bo["reg_alpha"], 1), 0),
        "colsample_bytree": max(min(lgb_bo["colsample_bytree"], 1), 0),
        "subsample": max(min(lgb_bo["subsample"], 1), 0),
        "num_leaves": int(round(lgb_bo["num_leaves"])),
        "min_child_samples": int(round(lgb_bo["min_child_samples"])),
    }

    with open("../../res/fea_lgb_bayesian1.pkl", "wb") as f:
        pickle.dump(lgb_params, f)

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
    with open("../../res/fea_cat_bayesian1.pkl", "wb") as f:
        pickle.dump(cat_params, f)

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
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": xgb_bo["learning_rate"],
        "gamma": xgb_bo["gamma"],
        "subsample": xgb_bo["subsample"],
        "colsample_bytree": xgb_bo["colsample_bytree"],
        "max_depth": int(round(xgb_bo["max_depth"])),
        "min_child_weight": int(round(xgb_bo["min_child_weight"])),
    }

    with open("../../res/fea_xgb_bayesian1.pkl", "wb") as f:
        pickle.dump(xgb_params, f)
