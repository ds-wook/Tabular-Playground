from typing import Any, Dict, Tuple

import numpy as np
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score

from data.datasets import X_train, y_train


def lgb_roc_eval(
    num_leaves: float,
    max_depth: float,
    min_split_gain: float,
    min_child_weight: float,
) -> float:
    params = {
        "n_estimators": 10000,
        "objective": "regression",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
        "learning_rate": 0.05,
        "num_leaves": int(round(num_leaves)),
        "max_depth": int(round(max_depth)),
        "min_split_gain": max(min(min_split_gain, 1), 0),
        "min_child_weight": max(min(min_child_weight, 1), 0),
    }
    model = LGBMRegressor(**params)
    scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    rmse_score = np.sqrt(-scores)
    return -np.mean(rmse_score)


def lgb_parameter(func: Any, params: Dict[str, Tuple[float]]) -> Dict[str, float]:
    lgbm_bo = BayesianOptimization(f=func, pbounds=params)
    lgbm_bo.maximize(init_points=5, n_iter=5)
    print("Optimization params:", lgbm_bo.max["params"])
    return lgbm_bo.max["params"]
