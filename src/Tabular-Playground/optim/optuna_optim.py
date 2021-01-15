import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from data.datasets import X, y


def xgb_objective(trial: optuna.Trial) -> float:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    params = {
        "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        "subsample": trial.suggest_categorical(
            "subsample", [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
        ),
        "n_estimators": 4000,
        "max_depth": trial.suggest_categorical(
            "max_depth", [5, 7, 9, 11, 13, 15, 17, 20]
        ),
        "random_state": trial.suggest_categorical("random_state", [24, 48, 2020]),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
    }
    model = XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=False,
    )
    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    return rmse


def lgb_objective(trial: optuna.Trial) -> float:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    params = {
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 100, 200),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 1),
        "min_child_samples": trial.suggest_int("min_child_samples", 4, 80),
        "learning_rate": trial.suggest_float("learning_rate", 0.0155, 0.5),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_uniform("subsample", 0.4, 1.0),
    }
    model = LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=False,
    )
    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    return rmse
