import pandas as pd
import sklearn.model_selection

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple

from IMLearn.utils import split_train_test
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X = np.load("C:\\Users\\X240\\IML.HUJI\\datasets\\" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        X_data = X[:, :-1]
        y_data = X[:, -1]

        def loss_callback(fit: Perceptron, x: np.ndarray, y: int):
            # use loss function on X, y
            losses.append(fit.loss(X_data, y_data))

        perceptron = Perceptron(callback=loss_callback)
        perceptron.fit(X=X_data, y=y_data)

        # Plot figure of loss as function of fitting iteration

        fig = go.Figure([go.Scatter(x=[i for i in range(1, len(losses) + 1)],
                                    y=losses, mode="markers")],
                        layout=go.Layout(
                            title="Losses over perceptron algorithm iterations - " + n,
                            xaxis_title="iteration number",
                            yaxis_title="loss"))
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X = np.load("C:\\Users\\X240\\IML.HUJI\\datasets\\" + f)

        # Fit models and predict over training set
        X_data = X[:, :-1]
        y_data = X[:, -1]
        train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(
            X_data, y_data)
        gaussian = GaussianNaiveBayes().fit(X=train_X, y=train_y)
        lda = LDA().fit(X=train_X, y=train_y)
        classes = np.unique(y_data)
        gaussian_predict = gaussian.predict(test_X)
        lda_predict = lda.predict(test_X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        gnb_accuracy = accuracy(test_y, gaussian_predict)
        lda_accuracy = accuracy(test_y, lda_predict)
        symbols = np.array(["circle", "x", "diamond"])
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Gaussian Naive Bayes prediction accuracy: {gnb_accuracy}",f"LDA prediction accuracy: {lda_accuracy}"],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        fig.add_traces(
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                       showlegend=False,
                       marker=dict(color=lda_predict.astype(int),
                                   symbol=symbols[
                                       test_y.astype(
                                           int)])), rows=1, cols=2)
        fig.add_traces(
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                       showlegend=False,
                       marker=dict(
                           color=gaussian_predict.astype(int),
                           symbol=symbols[
                               test_y.astype(int)])),
            rows=1, cols=1)



        # Add traces for data-points setting symbols and colors


        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(
            go.Scatter(x=gaussian.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                       marker=dict(color="black", symbol="x", size=8),
                       showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=lda.mu_[:,0], y=lda.mu_[:,1], mode="markers",marker=dict(color="black", symbol="x", size=8), showlegend=False ),row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(lda.classes_)):
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1,
                          col=2)
            fig.add_trace(get_ellipse(gaussian.mu_[i],
                                      np.diag(gaussian.vars_[i])),
                          row=1, col=1)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
