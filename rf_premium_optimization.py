# -*- coding: utf-8 -*-
"""Premium optimization with random forest"""

import numpy as np
import pickle
from sklearn.externals import joblib


def load_predict_data():
    """Predicts sales for a specified output

    Args:
        premium (float) : premium offered

    Returns:
       model (sklearn.ensemble.RandomForestRegressor): predictor model
       X (numpy.matrix) : X matrix
    """

    # Read the model
    data = pickle.load(open('rf_data.dat', 'rb'))
    X = data['X1']

    model = joblib.load('random_forest.pkl')

    return model, X


def predict_data(premium, model, X, select_best=0):
    """Predicts sales for a specified output

    Args:
        premium (float) : premium offered
        model (keras.model): predictor model
        scaler (sklearn.preprocessing.StandardScaler) : Scaler to transform variables
        X (numpy.matrix) : X matrix
        select_best (int) : if an integer > 0, select the best clients

    Returns:
        expected_income (float) : Expected income per customer
        expected_semesters_paid (float) : Expected number of semesters paid
        expected_sales (float) : Expected sales probability
        y (np.array) : Array with individual sales (0, 1) for each customer
    """
    # Fill the premium column
    X[:, 0] = premium

    # Predict X data
    y = model.predict(X)

    if select_best > 0:
        y = np.sort(y)
        y = y[-select_best:]

    expected_sales = y.mean()
    expected_income = expected_sales * premium

    return expected_sales, expected_income, y


if __name__ == "__main__":
    model, X = load_predict_data()
    print('Random clients')
    for premium in np.arange(11, 22, 0.5):
        expected_sales, expected_income, y = predict_data(premium, model, X)
        print(premium, expected_income, expected_sales)

    print('Select 5000 best clients')
    for premium in np.arange(11, 22, 0.5):
        expected_sales, expected_income, y = predict_data(premium, model, X, select_best=5000)
        print(premium, expected_income, expected_sales)






