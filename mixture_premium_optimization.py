# -*- coding: utf-8 -*-
"""Premium optimization with neural network"""

import numpy as np
import pickle
from keras.models import model_from_json
from sklearn.externals import joblib


def load_predict_data():
    """Predicts sales for a specified output

    Args:
        premium (float) : premium offered

    Returns:
       nn (keras.model): neural network model
       rf (sklearn.ensemble.RandomForestRegressor): random forest model
       scaler (sklearn.preprocessing.StandardScaler) : Scaler to transform variables
       Xnn (numpy.matrix) : X matrix for neural network model
       Xrf (numpy.matrix) : X matrix for random forest model
    """

    # Read the nn model
    data = pickle.load(open('nn_data.dat', 'rb'))
    scaler = data['scaler']
    Xnn = data['X2']

    # load json and create model
    json_file = open('nn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    nn = model_from_json(loaded_model_json)
    # load weights into new model
    nn.load_weights('nn_model.h5')

    # Read the rf model
    data = pickle.load(open('rf_data.dat', 'rb'))
    Xrf = data['X2']
    rf = joblib.load('random_forest.pkl')

    return nn, rf, scaler, Xnn, Xrf


def predict_data(premium, nn, rf, scaler, Xnn, Xrf, select_best=0):
    """Predicts sales for a specified output

    Args:
        premium (float) : premium offered
        nn (keras.model): predictor model
        rf (sklearn.ensemble.RandomForestRegressor): random forest model
        scaler (sklearn.preprocessing.StandardScaler) : Scaler to transform variables
        Xnn (numpy.matrix) : X matrix for neural network model
        Xrf (numpy.matrix) : X matrix for random forest model
        select_best (int) : if an integer > 0, select the best clients

    Returns:
        expected_income (float) : Expected income per customer
        expected_semesters_paid (float) : Expected number of semesters paid
        expected_sales (float) : Expected sales probability
        y (np.array) : Array with individual sales (0, 1) for each customer
    """
    ## NN predict

    # Fill the premium column
    Xnn[:, 0] = premium

    # Predict X data
    Xnorm = scaler.transform(Xnn)
    ynn = nn.predict(Xnorm)
    ynn = np.array([y[0] for y in ynn])

    ## RF predict

    # Fill the premium column
    Xrf[:, 0] = premium

    # Predict X data
    yrf = rf.predict(Xrf)

    # Average models
    y = np.mean([ynn, yrf], axis=0)

    if select_best > 0:
        y = np.sort(y)
        y = y[-select_best:]

    expected_sales = y.mean()
    expected_income = expected_sales * premium

    return expected_sales, expected_income, y


if __name__ == "__main__":
    nn, rf, scaler, Xnn, Xrf = load_predict_data()
    print('Random clients')
    for premium in np.arange(11, 22, 0.5):
        expected_sales, expected_income, y = predict_data(premium, nn, rf, scaler, Xnn, Xrf)
        print(premium, expected_income, expected_sales)

    print('Select 5000 best clients')
    for premium in np.arange(11, 22, 0.5):
        expected_sales, expected_income, y = predict_data(premium, nn, rf, scaler, Xnn, Xrf, select_best=5000)
        print(premium, expected_income, expected_sales)




