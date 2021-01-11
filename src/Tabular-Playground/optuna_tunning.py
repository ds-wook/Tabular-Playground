import pickle

import optuna.integration.lightgbm as lgb
from sklearn.model_selection import train_test_split
from data.datasets import X, y

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_valid, label=y_valid)
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "boosting_type": "gbdt",
}
model = lgb.train(
    params, dtrain, valid_sets=[dval], verbose_eval=100, early_stopping_rounds=100
)

params = model.params

with open("../../res/lgb_optuna.pkl", "wb") as f:
    pickle.dump(params, f)
