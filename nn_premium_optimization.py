# -*- coding: utf-8 -*-
"""Premium optimization with neural network"""

import numpy as np
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
    X = data['X1']

    # load json and create model
    json_file = open('nn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('nn_model.h5')

    return model, scaler, X


def predict_data(premium, model, scaler, X):
    """Predicts sales for a specified output

    Args:
        premium (float) : premium offered
        model (keras.model): predictor model
        scaler (sklearn.preprocessing.StandardScaler) : Scaler to transform variables
        X (numpy.matrix) : X matrix

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

    expected_income = y.mean()
    semesters_paid = y / premium
    expected_semesters_paid = semesters_paid.mean()
    expected_sales = sum(semesters_paid > 0.5) / len(semesters_paid)

    return expected_income, expected_semesters_paid, expected_sales, y


if __name__ == "__main__":
    model, scaler, X = load_predict_data()
    for premium in range(50):
        expected_income, expected_semesters_paid, expected_sales, y = predict_data(premium, model, scaler, X)
        print(premium, expected_income, expected_semesters_paid, expected_sales)





