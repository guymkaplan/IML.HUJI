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
    train_err, val_err = np.ndarray((11,), dtype=object), np.ndarray((11,), dtype=object)
    for k in K:
        train_err[k], val_err[k] = cross_validate(PolynomialFitting(k), train_X, train_y, scoring=mean_square_error,
                                                  cv=5)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=K, y=train_err, mode="markers+lines",
                              name='train errors'))
    fig2.add_trace(go.Scatter(x=K, y=val_err, mode="markers+lines",
                              name='validation errors'))
    fig2.update_layout(
        title="Q2: Train & Validation errors as a function of Polynomial Degree",
        xaxis_title="Polynomial Degree",
        yaxis_title="Train & Val errors")
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
    X_train, y_train, X_test, y_test = X[:n_samples, :], y[:n_samples], X[n_samples:, :], y[n_samples:]



    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lams = np.linspace(start=0.01, stop=2, num=n_evaluations)
    train_err_ridge, val_err_ridge, train_err_lasso, val_err_lasso = np.ndarray((n_evaluations,)), np.ndarray(
        (n_evaluations,)), np.ndarray((n_evaluations,)), np.ndarray((n_evaluations,))
    for l in range(len(lams)):
        train_err_ridge[l], val_err_ridge[l] = cross_validate(RidgeRegression(lam=lams[l]), X_train, y_train,
                                                              scoring=mean_square_error)
        train_err_lasso[l], val_err_lasso[l] = cross_validate(Lasso(alpha=lams[l]), X_train, y_train,
                                                              scoring=mean_square_error)

    fig2 = make_subplots(rows=1, cols=2,
                         subplot_titles=["Ridge Regression", "Lasso Regression"],
                         horizontal_spacing=0.01, vertical_spacing=.03)





    fig2.add_traces([go.Scatter(x=lams, y=train_err_ridge, mode="markers+lines", name="Ridge regression train error"),
                 go.Scatter(x=lams, y=val_err_ridge,
                            mode="markers+lines", name="Ridge regression validation error")], rows = 1, cols = 1)

    fig2.add_traces([go.Scatter(x=lams, y=train_err_lasso, mode="markers+lines", name="Lasso regression train error"),
                 go.Scatter(x=lams, y=val_err_lasso,
                            mode="markers+lines",name="Lasso regression validation error")], rows = 1, cols = 2)


    fig2.show()
    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = lams[int(np.argmin(val_err_ridge))]
    best_lasso = lams[int(np.argmin(val_err_lasso))]

    print(f"Best performing ridge lambda: {best_ridge}")
    print(f"Best performing lasso lambda: {best_lasso}")

    ridge_model_mse = RidgeRegression(lam=best_ridge).fit(X_train, y_train).loss(X_test, y_test)
    lasso_model = Lasso(alpha=best_lasso)
    lasso_model.fit(X_train,y_train)
    lasso_model_mse = mean_square_error(lasso_model.predict(X_test), y_test)
    regression_mse = LinearRegression().fit(X_train, y_train).loss(X_test, y_test)
    print(f"Ridge test error using best lambda: {ridge_model_mse}")
    print(f"Lasso test error using best lambda: {lasso_model_mse}")
    print(f"Least Squares test error: {regression_mse}")




if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(n_samples=100,noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()