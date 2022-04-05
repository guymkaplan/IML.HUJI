import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df.fillna(0, inplace=True)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    # looking at the possible cities, it is more likely that there is an error
    # in sampling than temp < -50 or temp > 50 celsius:
    df = df.drop(df[df['Temp'] < -50].index)
    df = df.drop(df[df['Temp'] > 50].index)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("C:/Users/X240/IML.HUJI/datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    X_israel = X[X['Country'] == "Israel"]
    fig1 = px.scatter(X_israel, x="DayOfYear", y="Temp", color=X_israel["Year"].astype(str),
                      title="Temperatures recorded in Israel, years 1995-2007")
    fig1.show()

    by_month = X_israel.groupby(['Month'])
    months_stds = by_month.agg('std')

    fig2 = px.bar(months_stds, y='Temp',
                  title="Standard deviation of temperatures per day, by month")
    fig2.show()


    # Question 3 - Exploring differences between countries
    by_country_month = X.groupby(['Country', 'Month'])

    average_temp = by_country_month.agg('mean')
    std_temp = by_country_month.agg('std')
    fig3 = px.line(average_temp, x=average_temp.axes[0].get_level_values(1),
                   y='Temp', color=average_temp.axes[0].get_level_values(0),
                   error_y=std_temp['Temp'], title="Average temperatures by country, per month, with standard devations as error bars")
    fig3.update_layout(xaxis_title="Month")
    fig3.show()



    # Question 4 - Fitting model for different values of `k`
    X_train, y_train, X_test, y_test = split_train_test(X_israel, X_israel['Temp'], train_proportion=0.75)
    k_ = np.array([k for k in range(1, 11)])
    losses = np.array([])
    for k in k_:
        model = PolynomialFitting(k).fit(X_train['DayOfYear'], y_train)
        loss = model.loss(X_test['DayOfYear'], y_test)
        losses = np.append(losses, loss)
        print(np.round(loss, 2))
    fig4 = px.bar(x = k_, y=losses, title="Polynomial fitting model MSE loss as a function of k-degree")
    fig4.update_layout(xaxis_title="k", yaxis_title="MSE")
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    polynomial_fitting = PolynomialFitting(5).fit(X_israel['DayOfYear'], X_israel['Temp'])
    countries = ['Jordan', 'South Africa', 'The Netherlands']
    losses = np.array([])
    for country in countries:
        loss = polynomial_fitting.loss(X[X['Country'] == country]['DayOfYear'],
                                X[X['Country'] == country]['Temp'])
        losses = np.append(losses, loss)

    fig5 = px.bar(x=countries, y=losses, title="Error of Israeli model for each country in database")
    fig5.update_layout(xaxis_title="Country", yaxis_title="Error")
    fig5.show()



