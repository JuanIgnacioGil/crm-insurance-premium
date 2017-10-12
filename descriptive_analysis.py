# -*- coding: utf-8 -*-
"""Functions for data analysis and plots"""

import pandas as pd
import numpy as np

import plotly.graph_objs as go

from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder


def analyze_feature(var, db1, categorical=False):

    """Analyzes a feature, draws a plot, and performs a chi square test

    Args:
        var (str): Feature name
        db1 (pandas.DataFrame): data
        db1_train (pandas.Index): indexes of the train set
        db1_test (pandas.Index): indexes of the test set

    Returns:
        chi (float): chi-square
        pval (float): p-value
        fig (plotly.graph_objs.Figure): plotly figure to be plot in a Jupyter notebook
    """

    # Logistic regression
    if categorical:
        chi, pval = chi2_categorical(var, db1)
    else:
        chi, pval = chi2_floating(var, db1)

    # Bar plot
    sales_average = db1['Sales'].mean()
    g = db1[['Sales', var]].groupby(var)
    data = pd.concat([g.mean(), np.sqrt((g.mean() * (1 - g.mean())) / g.count()),
                      100 * g.count() / g.count().sum()], axis=1)
    data.columns = ['Sales probability', 'standard error', 'Percentage of clients']

    dp = [
        # Chart
        go.Bar(
            x=data.index,
            y=data['Sales probability'],
            text=['{:.0f}% of clients'.format(x) for x in data['Percentage of clients']],
            name='Sales probability',
            error_y=dict(
                type='data',
                array=data['standard error'],
                visible=False)
        ),

        go.Scatter(
            x=[data.index[0], data.index[-1]],
            y=[sales_average, sales_average],
            mode='lines',
            name='Average: {:.3f}'.format(sales_average),
        )
    ]

    layout = go.Layout(
        barmode='group',
        title='{} (chi square:{:.0f}, p-value:{:.3f})'.format(var, chi, pval),
        yaxis=dict(title='Probability of sales'),
        xaxis=dict(title=var),
    )

    fig = go.Figure(data=dp, layout=layout)

    return chi, pval, fig


def chi2_floating(var, db1):
    """chi2 test for floating features

    Args:
        var (str): Feature name
        db1 (pandas.DataFrame): data
        db1_train (pandas.Index): indexes of the train set
        db1_test (pandas.Index): indexes of the test set

    Returns:
        c (float): chi-square
        pval (float): p-value
    """

    c, pval = chi2(db1[var].as_matrix().reshape(-1, 1), db1['Sales'].as_matrix())

    return c[0], pval[0]


def chi2_categorical(var, db1):
    """chi2 test for categorical features

    Args:
        var (str): Feature name
        db1 (pandas.DataFrame): data
        db1_train (pandas.Index): indexes of the train set
        db1_test (pandas.Index): indexes of the test set

    Returns:
        c (float): chi-square
        pval (float): p-value
    """

    le = LabelEncoder()
    X = le.fit_transform(db1[var]).reshape(-1, 1)

    c, pval = chi2(X, db1['Sales'].as_matrix())

    return c[0], pval[0]


def analyze_output(var, db1):

    """Analyzes an output feature:

    Args:
        var (str): Feature name
        db1 (pandas.DataFrame): data

    Returns:

        fig (plotly.graph_objs.Figure): plotly figure to be plot in a Jupyter notebook
    """

    # Histogram

    dp = [
        # Chart
        go.Histogram(
            x=db1[var],
            name=var,
        ),
    ]

    layout = go.Layout(
        barmode='group',
        yaxis=dict(title='Number of clients'),
        xaxis=dict(title=var),
    )

    fig = go.Figure(data=dp, layout=layout)

    return fig


def analyze_geo_feature(var, db1, categorical=False):

    """Analyzes a geographical feature, draws a Choropleth map, and returns the chi-square and p-value:

    Args:
        var (str): Feature name
        db1 (pandas.DataFrame): data
        db1_train (pandas.Index): indexes of the train set
        db1_test (pandas.Index): indexes of the test set

    Returns:
        score (float): Score of the logistic regression
        fig (plotly.graph_objs.Figure): plotly figure to be plot in a Jupyter notebook
    """

    scl = [[0.0, 'rgb(242,240,247)'], [0.2, 'rgb(218,218,235)'], [0.4, 'rgb(188,189,220)'], \
           [0.6, 'rgb(158,154,200)'], [0.8, 'rgb(117,107,177)'], [1.0, 'rgb(84,39,143)']]

    # Chi square and p-value
    if categorical:
        chi, pval = chi2_categorical(var, db1)
    else:
        chi, pval = chi2_floating(var, db1)

    # Bar plot
    sales_average = db1['Sales'].mean()
    g = db1[['Sales', var]].groupby(var)
    data = pd.concat([g.mean(), np.sqrt((g.mean() * (1 - g.mean())) / g.count()),
                      100 * g.count() / g.count().sum()], axis=1)
    data.columns = ['Sales probability', 'standard error', 'Percentage of clients']

    dp = [dict(
        type='choropleth',
        colorscale=scl,
        autocolorscale=False,
        locations=data.index,
        z=data['Sales probability'].astype(float),
        locationmode='Spain',
        marker=dict(
            line=dict(
                color='rgb(255,255,255)',
                width=2
            )),
        colorbar=dict(
            title="Sales probability")
    )]

    layout = go.Layout(
        barmode='group',
        title='{} (chi square:{:.0f}, p-value:{:.3f})'.format(var, chi, pval),
        yaxis=dict(title='Probability of sales'),
        xaxis=dict(title=var),
    )

    fig = go.Figure(data=dp, layout=layout)

    return chi, pval, fig
