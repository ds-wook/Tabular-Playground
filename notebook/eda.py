# %%
import pandas as pd

submission = pd.read_csv(
    "../input/tabular-playground-series-jan-2021/sample_submission.csv"
)
lgbm_preds = pd.read_csv('../res/lgbm_bayesian.csv')
tabnet_preds = pd.read_csv('../res/tabnet.csv')
submission['target'] = 0.5 * lgbm_preds['target'] + 0.5 * tabnet_preds['target']
submission.to_csv('../res/deep_ensemble.csv', index=False)


# %%
