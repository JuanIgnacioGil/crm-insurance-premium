# -*- coding: utf-8 -*-
"""Functions for data analysis and plots"""

import pandas as pd
import numpy as np

import plotly.graph_objs as go

from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder


def analyze_feature(var, db1, categorical=False):

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
    if categorical:
        score = logistic_categorical(var, db1)
    else:
        score = logistic_floating(var, db1)

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

    layout = go.Layout(barmode='group', title='{}. p-value: {}'.format(var, score),
                       yaxis=dict(title='Probability of sales'),
                        xaxis = dict(title=var))
    fig = go.Figure(data=dp, layout=layout)

    return score, fig


def logistic_floating(var, db1):
    """Fits a logistic regression and returns the score:

    Args:
        var (str): Feature name
        db1 (pandas.DataFrame): data
        db1_train (pandas.Index): indexes of the train set
        db1_test (pandas.Index): indexes of the test set

    Returns:
        score (float): Score of the logistic regression
    """

    c, pval = chi2(db1[var].as_matrix().reshape(-1, 1), db1['Sales'].as_matrix())

    return c[0], pval[0]


def logistic_categorical(var, db1):
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
    X = pd.get_dummies(db1[var], dummy_na=True).as_matrix()

    c, pval = chi2(X, db1['Sales'].as_matrix())

    return c, pval
