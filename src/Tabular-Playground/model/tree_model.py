from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def kfold_model(
    model: Any, n_fold: int, train: pd.DataFrame, target: pd.Series, test: pd.DataFrame
) -> Tuple[np.ndarray, float]:
    folds = KFold(n_splits=n_fold, random_state=48, shuffle=True)
    splits = folds.split(train, target)
    y_preds = np.zeros(test.shape[0])
    oof_preds = np.zeros(train.shape[0])
    for fold_n, (train_index, valid_index) in tqdm(enumerate(splits)):
        model_name = model.__class__.__name__
        print(f"\t{model_name} Learning Start!")
        print(f"Fold: {fold_n + 1}")
        X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        evals = [(X_train, y_train), (X_valid, y_valid)]
        if model_name == "LGBMRegressor":
            model.fit(
                X_train,
                y_train,
                eval_set=evals,
                eval_metric="rmse",
                early_stopping_rounds=100,
                verbose=100,
            )
            oof_preds[valid_index] = model.predict(X_valid)
            y_preds += model.predict(test) / n_fold
        else:
            model.fit(
                X_train,
                y_train,
                eval_set=evals,
                early_stopping_rounds=100,
                verbose=100,
            )
            oof_preds[valid_index] = model.predict(X_valid)
            y_preds += model.predict(test) / n_fold
        del X_train, X_valid, y_train, y_valid
    scores = mean_squared_error(target, oof_preds, squared=False)
    print(f"OOF Score: {scores:.5f}")
    return y_preds, scores
