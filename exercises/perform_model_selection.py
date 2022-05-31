from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # f(x) = (x^2 - 4)(x^2 - 1)(x + 3)
    # and split into training- and testing portions
    x_arr = np.random.uniform(-1.2, 2, n_samples)
    epsilons = np.random.normal(0, noise, n_samples)

    def f(x):
        return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    y = np.vectorize(f)(x_arr) + epsilons
    train_X, train_y, test_X, test_y = split_train_test(X=pd.DataFrame(x_arr), y=pd.Series(y), train_proportion=2 / 3)

    train_X = train_X.to_numpy().reshape((train_X.shape[0],))
    train_y = train_y.to_numpy().reshape((train_y.shape[0],))
    test_X = test_X.to_numpy().reshape((test_X.shape[0],))
    test_y = test_y.to_numpy().reshape((test_y.shape[0],))

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=x_arr, y=f(x_arr), mode="markers",
                              name='true model'))

    fig1.add_trace(go.Scatter(x=train_X, y=train_y, mode="markers",
                              name='train set'))
    fig1.add_trace(go.Scatter(x=test_X, y=test_y, mode="markers",
                              name='test set'))
    fig1.update_layout(
        title="Q1: Dataset generation",
        xaxis_title="x",
        yaxis_title="f(x) + epsilon")

    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    K = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_err, val_err = np.ndarray((11,)), np.ndarray((11,))
    for k in K:
        train_err[k], val_err[k] = cross_validate(PolynomialFitting(k), train_X, train_y, scoring=mean_square_error,
                                                  cv=5)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=K, y=train_err, mode="markers+lines",
                              name='train errors'))
    fig2.add_trace(go.Scatter(x=K, y=val_err, mode="markers+lines",
                              name='validation errors'))
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = int(np.argmin(val_err))
    estimator = PolynomialFitting(k_star).fit(train_X, train_y)
    print(f"Value of best scoring k: {k_star}")
    print(f"Test error for k_star: {np.round(estimator.loss(test_X, test_y), 2)}")
    print(f"Validation error for k_star: {np.round(val_err[k_star], 2)}")


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(train_size=n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lams = np.linspace(start=0, stop=50, num=n_evaluations)
    train_err_ridge, val_err_ridge, train_err_lasso, val_err_lasso = np.ndarray((n_evaluations,)), np.ndarray(
        (n_evaluations,)), np.ndarray((n_evaluations,)), np.ndarray((n_evaluations,))
    for l in lams:
        train_err_ridge[l], val_err_ridge[l] = cross_validate(RidgeRegression(lam=l), X_train, y_train,
                                                              scoring=mean_square_error)
        train_err_lasso[l], val_err_lasso[l] = cross_validate(Lasso(alpha=l), X_train, y_train,
                                                              scoring=mean_square_error)

        # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
