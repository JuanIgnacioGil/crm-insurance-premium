# -*- coding: utf-8 -*-
"""Premium optimization with neural network"""

import numpy as np
import pandas as pd
import pickle
from keras.models import model_from_json


def load_predict_data():
    """Predicts sales for a specified output

    Args:
        premium (float) : premium offered

    Returns:
       model (keras.model): predictor model
       scaler (sklearn.preprocessing.StandardScaler) : Scaler to transform variables
       X (numpy.matrix) : X matrix
    """

    # Read the model
    data = pickle.load(open('nn_data.dat', 'rb'))
    scaler = data['scaler']
    X = data['X2']

    # load json and create model
    json_file = open('nn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('nn_model.h5')

    return model, scaler, X


def predict_data(premium, model, scaler, X, select_best=0):
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
    Xnorm = scaler.transform(X)
    y = model.predict(Xnorm)
    y = np.array([c[0] for c in y])

    if select_best > 0:
        y = np.sort(y)
        y = y[-select_best:]

    expected_sales = y.mean()
    expected_income = expected_sales * premium

    return expected_sales, expected_income, y


def predict_premium_vector(min_premium, max_premium, step, select_best=0):
    """Predicts sales for different premiums

    Args:
        min_premium (float) : minimum premium offered
        max_premium (float) : maximum premium offered
        step (float) : step between two premiums
        select_best (int) : if an integer > 0, select the best clients
    Returns:
        pandas.DataFrame
    """
    model, scaler, X = load_predict_data()
    premium_range = np.arange(min_premium, max_premium, step)
    data = pd.DataFrame(index=premium_range, columns=['Sales', 'Income'])

    for p in np.arange(min_premium, max_premium, step):
        sales, income, _ = predict_data(p, model, scaler, X, select_best=select_best)
        data.loc[p] = {'Sales': sales, 'Income': income}

    return data


if __name__ == "__main__":
    data = predict_premium_vector(11.12, 21.85, 0.01, select_best=0)
    print(data)