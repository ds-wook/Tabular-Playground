from typing import Any, Dict, Tuple

import numpy as np
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score

from data.datasets import X, y


def lgb_rmse_eval(
    learning_rate: float,
    reg_lambda: float,
    reg_alpha: float,
    num_leaves: float,
    min_child_samples: float,
) -> float:
    params = {
        "n_estimators": 10000,
        "objective": "regression",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": max(min(learning_rate, 1), 0),
        "reg_lambda": max(min(reg_lambda, 1), 0),
        "reg_alpha": max(min(reg_alpha, 1), 0),
        "num_leaves": int(round(num_leaves)),
        "min_child_samples": int(round(min_child_samples)),
    }
    model = LGBMRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse_score = np.sqrt(-scores)
    return -np.mean(rmse_score)


def lgb_parameter(func: Any, params: Dict[str, Tuple[float]]) -> Dict[str, float]:
    lgbm_bo = BayesianOptimization(f=func, pbounds=params)
    lgbm_bo.maximize(init_points=5, n_iter=20)
    print("Optimization params:", lgbm_bo.max["params"])
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
        "learning_rate": max(min(learning_rate, 1), 0),
        "l2_leaf_reg": max(min(l2_leaf_reg, 1), 0),
    }
    model = CatBoostRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse_score = np.sqrt(-scores)
    return -np.mean(rmse_score)


def cat_parameter(func: Any, params: Dict[str, Tuple[float]]) -> Dict[str, float]:
    cat_bo = BayesianOptimization(f=func, pbounds=params)
    cat_bo.maximize(init_points=5, n_iter=10)
    print("Ooptimization params:", cat_bo.max["params"])
    return cat_bo.max["params"]
