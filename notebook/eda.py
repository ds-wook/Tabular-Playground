# %%
import pandas as pd

bayesian_params = pd.read_pickle("../res/lgb_bayesian.pkl")
# %%
optuna_params = pd.read_pickle("../res/lgb_optuna.pkl")
# %%
print(bayesian_params)
print(optuna_params)
# %%
submission = pd.read_csv(
    "../input/tabular-playground-series-jan-2021/sample_submission.csv"
)

lgb_bayesian = pd.read_csv("../res/lgbm_bayesian.csv")
lgb_optuna = pd.read_csv("../res/lgb_optuna.csv")

submission["target"] = 0.5 * lgb_bayesian["target"] + 0.5 * lgb_optuna["target"]
submission.to_csv("../res/hyper_ensemble.csv", index=False)

# %%
