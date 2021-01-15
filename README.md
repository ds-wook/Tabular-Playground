# Tabular-Playground

This is a collection of my code from the [Tabular Playground Series - Jan 2021](https://www.kaggle.com/c/tabular-playground-series-jan-2021) Kaggle competition.

## Code Style
I follow [black](https://pypi.org/project/black/) for code style. Black is a PEP 8 compliant opinionated formatter.

#### Non-FE Hyper Parameter Tunning
|method|baseline|OOF(5-fold)|Public LB|Private LB|
|------|:------:|:---------:|:-------:|:--------:|
|LGBM bayesian_optim|0.6956|0.69664|0.69839|-|
|XGB bayesian_optim|0.6963|-|-|-|
|Cat bayesian_optim|-|-|-|-|
|LGBM optuna|0.6988|0.69756|0.69831|-|

#### FE Hyper Parameter Tunning
|method|baseline|OOF(5-fold)| Public LB|Private LB|
|------|:------:|:---------:|:--------:|:--------:|
|LGBM bayesian_optim|-|-|-|-|
|Cat bayesian_optim|-|-|-|-|
|LGBM optuna|-|-|-|-|
|XGB optuna|-|-|-|-|
