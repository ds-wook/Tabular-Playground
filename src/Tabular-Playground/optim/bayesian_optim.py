from typing import Any, Dict, Tuple

import numpy as np
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from data.datasets import X, y, X_test
from model.tree_model import kfold_model


def lgb_rmse_eval(
    learning_rate: float,
    max_depth: float,
    reg_lambda: float,
    reg_alpha: float,
    num_leaves: float,
    colsample_bytree: float,
    subsample: float,
    min_child_samples: float,
) -> float:
    params = {
        "n_estimators": 20000,
        "objective": "regression",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": max(min(learning_rate, 1), 0),
        "max_depth": int(round(max_depth)),
        "reg_lambda": max(min(reg_lambda, 1), 0),
        "reg_alpha": max(min(reg_alpha, 1), 0),
        "colsample_bytree": max(min(colsample_bytree, 1), 0),
        "subsample": max(min(subsample, 1), 0),
        "num_leaves": int(round(num_leaves)),
        "min_child_samples": int(round(min_child_samples)),
    }
    model = LGBMRegressor(**params)
    preds, scores = kfold_model(model, 5, X, y, X_test)
    return -scores


def lgb_parameter(func: Any, params: Dict[str, Tuple[float]]) -> Dict[str, float]:
    lgbm_bo = BayesianOptimization(f=func, pbounds=params)
    lgbm_bo.maximize(init_points=5, n_iter=35)
    return lgbm_bo.max["params"]


def cat_rmse_eval(
    depth: float, bagging_temperature: float, l2_leaf_reg: float, learning_rate: float
) -> float:
    params = {
        "iterations": 100,
        "loss_function": "RMSE",
        "verbose": False,
        "depth": int(round(depth)),
        "bagging_temperature": max(min(bagging_temperature, 1), 0),
        "learning_rate": learning_rate,
        "l2_leaf_reg": l2_leaf_reg,
    }
    model = CatBoostRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse_score = np.sqrt(-scores)
    return -np.mean(rmse_score)


def cat_parameter(func: Any, params: Dict[str, Tuple[float]]) -> Dict[str, float]:
    cat_bo = BayesianOptimization(f=func, pbounds=params)
    cat_bo.maximize(init_points=5, n_iter=25)
    return cat_bo.max["params"]


def xgb_rmse_eval(
    gamma: float,
    max_depth: float,
    min_child_weight: float,
) -> float:
    params = {
        "n_estimators": 10000,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 1.0,
        "gamma": gamma,
        "max_depth": int(round(max_depth)),
        "min_child_weight": int(round(min_child_weight)),
    }
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse_score = np.sqrt(-scores)
    return -np.mean(rmse_score)


def xgb_parameter(func: Any, params: Dict[str, Tuple[float]]) -> Dict[str, float]:
    xgb_bo = BayesianOptimization(f=func, pbounds=params)
    xgb_bo.maximize(init_points=5, n_iter=10)
    return xgb_bo.max["params"]
