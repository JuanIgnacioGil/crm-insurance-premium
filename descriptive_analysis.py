# -*- coding: utf-8 -*-
"""Functions for data analysis and plots"""

import pandas as pd
import numpy as np

import plotly.graph_objs as go

from sklearn.linear_model import LogisticRegression


def analyze_feature(var, db1, db1_train, db1_test, categorical=False):

    """Analyzes a feature, draws a plot, fits a logistic regression and returns the score:

    Args:
        var (str): Feature name
        db1 (pandas.DataFrame): data
        db1_train (pandas.Index): indexes of the train set
        db1_test (pandas.Index): indexes of the test set

    Returns:
        score (float): Score of the logistic regression
        fig (plotly.graph_objs.Figure): plotly figure to be plot in a Jupyter notebook
    """

    # Logistic regression
    score = logistic_floating(var, db1, db1_train, db1_test)

    # Bar plot
    g = db1[['Sales', var]].groupby(var)
    data = pd.concat([g.mean(), np.sqrt((g.mean() * (1 - g.mean())) / g.count()), g.count()], axis=1)
    data.columns = ['Sales probability', 'standard error', 'Number of clients']

    dp = [go.Bar(
        x=data.index,
        y=data['Sales probability'],
        name='Sales probability',
        error_y=dict(
            type='data',
            array=data['standard error'],
            visible=True)
    )]

    layout = go.Layout(barmode='group', title='{}. Score {}'.format(var, score))
    fig = go.Figure(data=dp, layout=layout)

    return score, fig


def logistic_floating(var, db1, db1_train, db1_test):
    """Fits a logistic regression and returns the score:

    Args:
        var (str): Feature name
        db1 (pandas.DataFrame): data
        db1_train (pandas.Index): indexes of the train set
        db1_test (pandas.Index): indexes of the test set

    Returns:
        score (float): Score of the logistic regression
    """
    lr = LogisticRegression()

    lr.fit(db1.loc[db1_train, var].as_matrix().reshape(-1, 1),
           db1.loc[db1_train, 'Sales'].as_matrix())

    score = lr.score(db1.loc[db1_test, var].as_matrix().reshape(-1, 1),
                     db1.loc[db1_test, 'Sales'].as_matrix())

    return score


def logistic_categorical(var, db1, db1_train, db1_test):
    """Analyzes a feature, draws a plot, fits a logistic regression and returns the score:

    Args:
        var (str): Feature name
        db1 (pandas.DataFrame): data
        db1_train (pandas.Index): indexes of the train set
        db1_test (pandas.Index): indexes of the test set

    Returns:
        score (float): Score of the logistic regression
        fig (plotly.graph_objs.Figure): plotly figure to be plot in a Jupyter notebook
    """
    X = pd.get_dummies(db1[var])
    X_train = X[db1_train].as_matrix()
    X_test = X[db1_test].as_matrix()

    lr = LogisticRegression()
    lr.fit(db1.loc[db1_train, var].as_matrix().reshape(-1, 1),
           db1.loc[db1_train, 'Sales'].as_matrix())

    score = lr.score(db1.loc[db1_test, var].as_matrix().reshape(-1, 1),
                     db1.loc[db1_test, 'Sales'].as_matrix())

    return score
