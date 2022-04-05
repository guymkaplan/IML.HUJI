from sklearn.model_selection import ParameterGrid

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    # df.fillna(0, inplace=True)
    df = df.dropna().drop_duplicates()


    df = pd.get_dummies(df, columns=['zipcode'], drop_first=True)

    df.drop(df[df['price'] == 0].index, inplace= True)
    df.drop(df[df['price'] == "nan"].index, inplace= True)
    df.drop(df[df['sqft_living'] < df['bedrooms'] * 150].index, inplace=True)
    y = pd.Series(df['price'])
    y[y < 0] = -y


    df.drop(['price', 'id', 'date', 'sqft_living', 'long'], axis=1,inplace=True)
    df = np.abs(df)

    df['yr_renovated'] = df[['yr_renovated', 'yr_built']].max(axis=1)
    return (df, y)


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_sigma = np.sqrt(np.sum((y - y.mean()) ** 2))
    for (feature_name, x) in X.iteritems():
        corr = (x - x.mean()).dot(y - y.mean()) / (
                    np.sqrt(np.sum((x - x.mean()) ** 2)) * y_sigma)
        fig = go.Figure([go.Scatter(x=x, y=y, mode='markers')],
                        layout=go.Layout(
                            title=feature_name + " Pearson Correlation: " + str(
                                corr), xaxis_title=feature_name,
                            yaxis_title="the response"))
        fig.write_image(output_path + "/" + feature_name + ".png")



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("C:/Users/X240/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "G:/My Drive/Courses/2B/IML")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*stds, mean+2*stds)

    averages = []
    stds = []
    p_ = np.arange(10,101)
    for p in p_:
        p_averages = np.array([])
        for i in range(10):
            X, y, X2, y2 = split_train_test(train_X, train_y, train_proportion=p/100)
            model = LinearRegression()  # 2
            model.fit(X.to_numpy(), y.to_numpy())
            p_averages = np.append(p_averages, model.loss(test_X.to_numpy(), test_y.to_numpy()))  # 3
        # 4
        averages.append(p_averages.mean())
        stds.append(p_averages.std())
    stds = np.array(stds)
    averages = np.array(averages)
    p_for_graph = [str(p) + "%" for p in p_]
    fig = go.Figure([go.Scatter(x=p_for_graph, y=averages, mode='markers+lines',showlegend=False),
                     go.Scatter(x=p_for_graph, y=averages + (2 * stds), fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
                          go.Scatter(x=p_for_graph, y=averages - (2 * stds), fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False)],
                    layout=go.Layout(
                        title="Average loss of model in response to % of data used for training, with error ribbon of size (mean-2*stds, mean+2*stds)", xaxis_title="% of data used",
                        yaxis_title="average loss with confidence interval of (mean-2*stds, mean+2*stds)"))
    fig.show()





