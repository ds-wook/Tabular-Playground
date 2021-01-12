import optuna
from lightgbm import LGBMRegressor
import numpy as np
from data.datasets import X, y
from sklearn.model_selection import KFold


def lgbm_oof(trial: optuna.Trial) -> float:
    params = {
        "learning_rate": 0.005,
        "num_leaves": trial.suggest_int("num_leaves", 31, 100),
        "n_estimators": trial.suggest_categorical(
            "n_estimators", [250, 300, 350, 400, 450]
        ),
        "eta": trial.suggest_loguniform("eta", 1e-2, 1e-1),
        "max_depth": trial.suggest_categorical("max_depth", [6, 8, 10, 12]),
        "subsample": trial.suggest_discrete_uniform("subsample", 0.6, 1, 0.1),
        "colsample_bytree": trial.suggest_discrete_uniform(
            "colsample_bytree", 0.6, 1, 0.1
        ),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 11),
        "min_child_sample": trial.suggest_int("min_child_sample", 20, 50),
        "random_state": 42,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_error = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        train_X, val_X = X.iloc[train_idx, :], X.iloc[val_idx, :]
        train_y, val_y = y.iloc[train_idx], y.iloc[val_idx]

        lgbm_model = LGBMRegressor(**params)
        lgbm_model.fit(train_X, train_y)
        pred_y = lgbm_model.predict(val_X)
        in_fold_rmse = np.sqrt(np.mean((val_y - pred_y) ** 2))

        fold_error.append(in_fold_rmse)

    oof_rmse = np.sum(fold_error) / len(fold_error)
    return oof_rmse


def objective(trail: optuna.Trial) -> float:
    return lgbm_oof(trail)
