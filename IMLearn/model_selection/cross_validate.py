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
    vector = (X.ndim == 1)
    if vector:
        S = np.stack((X, y), axis=-1) # for vector
    else:
        S = np.concatenate((X, y), axis=1)  # for matrix

    sets = np.array_split(S, cv)
    train_errors = np.ndarray((cv,))
    val_errors = np.ndarray((cv,))
    for i in range(cv):
        current_sets = np.delete(sets, i, 0)
        if vector:
            current_X = np.concatenate(current_sets, axis=0)[:, 0]
            current_y = np.concatenate(current_sets, axis=0)[:, -1]
            model = estimator.fit(current_X, current_y)
            y_hat_dev = model.predict(sets[i][:, 0])
            y_true_dev = sets[i][:, -1]
            val_errors[i] = scoring(y_hat_dev, y_true_dev)
        else:
            current_X = np.concatenate(current_sets, axis=0)[:, :-1]
            current_y = np.concatenate(current_sets, axis=0)[:, -1]
            model = estimator.fit(current_X, current_y)
            y_hat_dev = model.predict(sets[i][:, :-1])
            y_true_dev = sets[i][:, -1]
            val_errors[i] = scoring(y_hat_dev, y_true_dev)
        # model = estimator.fit(current_X, current_y)

        y_hat_without_dev = model.predict(current_X)
        y_true_without_dev = current_y
        train_errors[i] = scoring(y_hat_without_dev, y_true_without_dev)
        # if vector:
        #     y_hat_dev = model.predict(sets[i][:, 0])
        #     y_true_dev = sets[i][:, -1]
        #     val_errors[i] = scoring(y_hat_dev, y_true_dev)
        # else:
        #     y_hat_dev = model.predict(sets[i][:,:-1])
        #     y_true_dev = sets[i][:, -1]
        #     val_errors[i] = scoring(y_hat_dev, y_true_dev)

    return float(np.mean(train_errors)), float(np.mean(val_errors))
