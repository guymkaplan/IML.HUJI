import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt

from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners import AdaBoost
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(wl=DecisionStump, iterations=n_learners)
    adaboost.fit(train_X, train_y)
    train_error = np.ndarray((n_learners,))
    test_error = np.ndarray((n_learners,))
    for i in range(n_learners):
        train_error[i] = adaboost.partial_loss(train_X, train_y, i + 1)
        test_error[i] = adaboost.partial_loss(test_X, test_y, i + 1)
    fig_q1 = go.Figure()

    fig_q1.add_trace(go.Scatter(x=np.arange(1, n_learners + 1), y=test_error, mode='markers+lines',
                              name='test set'))

    fig_q1.add_trace(go.Scatter(x=np.arange(1, n_learners + 1), y=train_error, mode="markers+lines",
                              name='train set'))
    fig_q1.update_layout(
        title="Test errors as a function of number of fitted learners",
        xaxis_title="# of fitted learners",
        yaxis_title="loss")
    fig_q1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    model_names = [f"Adaboost using {t} iterations" for t in T]


    fig_q2 = make_subplots(rows=2, cols=2,
                        subplot_titles=[rf"$\textbf{{{m}}}$" for m in
                                        model_names],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    symbols = np.array(["circle", "x"])

    def decision_surface(predict, xrange, yrange, t, density=120, dotted=False,
                         colorscale=custom, showscale=True):
        xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange,
                                                                    density)
        xx, yy = np.meshgrid(xrange, yrange)
        pred = predict(np.c_[xx.ravel(), yy.ravel()], t)

        if dotted:
            return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1,
                              mode="markers", marker=dict(color=pred, size=1,
                                                          colorscale=colorscale,
                                                          reversescale=False),
                              hoverinfo="skip", showlegend=False)
        return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape),
                          colorscale=colorscale, reversescale=False,
                          opacity=.7, connectgaps=True, hoverinfo="skip",
                          showlegend=False, showscale=showscale)

    for i, t in enumerate(T):
        fig_q2.add_traces(
            [decision_surface(adaboost.partial_predict, lims[0], lims[1],
                              showscale=False, t=t),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                        showlegend=False,
                        marker=dict(color=test_y, symbol=symbols[
                            np.where(test_y < 0, 0, test_y).astype(int)],
                                    colorscale=[custom[0],
                                                custom[-1]],
                                    line=dict(color="black",
                                              width=1)))],

            rows=(i // 2) + 1, cols=(i % 2) + 1)
        fig_q2.update_layout(
            title=rf"$\textbf{{Decision Boundaries of Adaboost with different number of iterations}}$",
            margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
    fig_q2.show()

    # Question 3: Decision surface of best performing ensemble

    best_performance = np.argmin(test_error)
    fig_q3 = go.Figure([decision_surface(adaboost.partial_predict, lims[0], lims[1],
                          showscale=False, t=best_performance),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(color=test_y, symbol=symbols[
                        np.where(test_y < 0, 0, test_y).astype(int)],
                                colorscale=[custom[0],
                                            custom[-1]],
                                line=dict(color="black",
                                          width=1)))])
    best_acc = accuracy(test_y, adaboost.partial_predict(test_X, int(best_performance)))
    fig_q3.update_layout(
        title=f"Decision Surface with Ensemble Size of {best_performance} and Accuracy of {best_acc}",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig_q3.show()


    # Question 4: Decision surface with weighted samples
    D = (adaboost.D_ / np.max(adaboost.D_)) * 10
    fig_q4 = go.Figure([decision_surface(adaboost.partial_predict, lims[0],
                                       lims[1], t=250, showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                                 mode="markers", showlegend=False,
                                 marker=dict(color=train_y, symbol=symbols[
                                     np.where(train_y < 0, 0, train_y).astype(
                                         int)], size=D
                                             ,
                                             colorscale=[custom[1], custom[2]],
                                             line=dict(color="black",
                                                       width=1)))])

    fig_q4.update_layout(title="Training set with point size proportional to its weight and color as label", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig_q4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0, n_learners=250, train_size=5000,
                              test_size=500)
    fit_and_evaluate_adaboost(noise=0.4, n_learners=250, train_size=5000,
                              test_size=500)
