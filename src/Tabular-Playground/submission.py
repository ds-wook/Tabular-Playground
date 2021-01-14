import pandas as pd

path = "../../input/tabular-playground-series-jan-2021/"
submission = pd.read_csv(path + "sample_submission.csv")

lgb_preds = pd.read_csv("../../res/lgb_bayesian1.csv")
xgb_preds = pd.read_csv("../../res/xgb_bayesian1.csv")

submission['target'] = 0.99 * lgb_preds["target"] + 0.01 * xgb_preds["target"]
submission.to_csv("../../res/ensemble_2boosting1.csv")
