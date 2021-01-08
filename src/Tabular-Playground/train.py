import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from tqdm import tqdm

from data.datasets import X, X_test, y, submission
from optim.bayesian_optim import lgb_roc_eval, lgb_parameter

np.seterr(divide="ignore", invalid="ignore")

bayesian_params = {
    "learning_rate": (0.0001, 0.05),
    "reg_lambda": (0, 1),
    "reg_alpha": (0, 1),
    "num_leaves": (100, 200),
    "min_child_samples": (20, 50),
}

lgb_bo = lgb_parameter(lgb_roc_eval, bayesian_params)

params = {
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

if __name__ == "__main__":
    n_fold = 5
    kf = KFold(n_splits=n_fold)
    oof = np.zeros(len(y))
    y_preds = np.zeros(len(X_test))

    for fold_n, (train_idx, test_idx) in tqdm(enumerate(kf.split(X, y))):
        print(f'{fold_n + 1} Fold Start it!')
        X_train, X_valid = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[test_idx]

        model = LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric="rmse",
            verbose=100,
            early_stopping_rounds=100,
        )
        y_preds += model.predict(X_test) / n_fold
        del X_train, X_valid, y_train, y_valid

    submission["target"] = y_preds
    submission.to_csv("../../res/baseline_submission.csv", index=False)
