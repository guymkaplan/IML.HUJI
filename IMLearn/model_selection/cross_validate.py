from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    X_splitted, y_splitted = np.array_split(X, cv), np.array_split(y, cv)
    score_train, score_val = np.ndarray((cv,), dtype=object), np.ndarray((cv,), dtype=object)

    for k in range(cv):
        train_data = np.concatenate(np.delete(X_splitted, k, axis=0), axis=0)
        labels = np.concatenate(np.delete(y_splitted, k, axis=0), axis=0)
        estimator_fitted = estimator.fit(train_data, labels)
        score_train[k] = scoring(labels, estimator_fitted.predict(train_data))
        score_val[k] = scoring(y_splitted[k], estimator_fitted.predict(X_splitted[k]))

    return float(np.mean(score_train)), float(np.mean(score_val))
