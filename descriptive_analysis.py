# -*- coding: utf-8 -*-
"""Functions for data analysis and plots"""

import pandas as pd
import numpy as np

import plotly.graph_objs as go

from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder


def analyze_feature(var, db1, categorical=False, continous=False):

    """Analyzes a feature, draws a plot, and performs a chi square test

    Args:
        var (str): Feature name
        db1 (pandas.DataFrame): data
        db1_train (pandas.Index): indexes of the train set
        db1_test (pandas.Index): indexes of the test set
        categorical (bool): True if the data is categorical (defaults to False)
        continous (bool): True if the data is continous (defaults to False)

    Returns:
        chi (float): chi-square
        pval (float): p-value
        fig (plotly.graph_objs.Figure): plotly figure to be plot in a Jupyter notebook
    """

    title = var
    sales_average = db1['Sales'].mean()

    # If the data is continous, cut it in buckets
    if continous:
        bins = sorted(list(set(db1[var].quantile(np.linspace(0, 1, num=6)))))
        db1['binned'] = pd.cut(db1[var], bins=bins, include_lowest=True)
        var = 'binned'
        categorical = True

    # Chi square test
    if categorical:
        chi, pval = chi2_categorical(var, db1)
    else:
        chi, pval = chi2_floating(var, db1)

    # Bar plot
    g = db1[['Sales', var]].groupby(var)
    data = pd.concat([g.mean(), g.std(), 100 * g.count() / g.count().sum()], axis=1)

    data.columns = ['Sales', 'standard error', 'Percentage of clients']

    dp = [
        # Chart
        go.Bar(
            x=data.index,
            y=data['Sales'],
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
        title='{} (chi square:{:.0f}, p-value:{:.3f})'.format(title, chi, pval),
        yaxis=dict(title='Sales probability', range=[0, 1]),
        xaxis=dict(title=title),
        paper_bgcolor='rgb(241,239,232)',
        plot_bgcolor='rgb(241,239,232)'
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

    ley = LabelEncoder()
    y = ley.fit_transform(db1['Sales'])

    c, pval = chi2(db1[var].as_matrix().reshape(-1, 1), y)

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

    leX = LabelEncoder()
    X = leX.fit_transform(db1[var]).reshape(-1, 1)

    ley = LabelEncoder()
    y = ley.fit_transform(db1['Sales'])

    c, pval = chi2(X, y)

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