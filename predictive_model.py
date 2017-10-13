# -*- coding: utf-8 -*-
"""Predictive model"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

# Features to use
features = ['ProdActive', 'ProdBought', 'NumberofCampaigns', 'Email', 'Province', 'TenureYears', 'Socieconomic Status',
            'Price Sensitivity', 'Right Address', 'PhoneType','Premium Offered', 'Estimated number of cars',
            'Living Area (m^2)', 'Number of Fixed Lines', 'yearBuilt', 'Credit', 'Probability of Second Residence']

def prepare_data():
    """Prepare data for analysis

    Args:

    Returns:
        X_train (np.matrix): train X
        X_test (np.matrix): test X
        y_train (np.matrix): train y
        y_test (np.matrix): test y
    """

    # Read data
    xls = pd.ExcelFile('Database.xlsx')
    db1 = xls.parse(1)

    X = proccess_X(db1)
    y = db1.loc[:, 'Number of Semesters Paid'].as_matrix()

    # Normalize X
    scaler = StandardScaler()
    Xnorm = scaler.fit_transform(X)

    # Train and test sets
    X_train, X_test, y_train, y_test = train_test_split(Xnorm, y, test_size=0.15, random_state=42)

    # Pickle the data
    data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'scaler': scaler}
    pickle.dump(data, open('ml_data.dat', 'wb'))

    return X_train, X_test, y_train, y_test, scaler


def proccess_X(db1):
    """Create X matrix (common for train and test data)

    Args:
        db1 (pandas.DataFrame)

    Returns:
        Xmat (np.matrix): X
    """

    X = pd.DataFrame()

    # Number of Semesters Paid
    var = 'Number of Semesters Paid'
    db1.loc[np.isnan(db1[var]), var] = 0
    X[var] = db1[var].copy()

    # ProdActive
    var = 'ProdActive'
    X[var] = db1[var].copy()

    # ProdBought
    var = 'ProdBought'
    X[var] = db1[var].copy()

    # NumberofCampaigns
    var = 'NumberofCampaigns'
    X[var] = db1[var].copy()

    # Email
    var = 'Email'
    X[var] = db1[var].copy()

    # Province
    var = 'Province'
    dummies = pd.get_dummies(db1[var], dummy_na=True)
    X = pd.concat([X, dummies], axis=1)

    # TenureYears
    var = 'TenureYears'
    db1[var] = 2013 - db1['Tenure']
    X[var] = db1[var].copy()

    # Socieconomic Status
    var = 'Socieconomic Status'
    db1.loc[pd.isnull(db1[var]), var] = 0
    db1.loc[db1[var] == 'Low', var] = 1
    db1.loc[db1[var] == 'Medium', var] = 2
    db1.loc[db1[var] == 'High', var] = 3
    db1.loc[db1[var] == 'Very High', var] = 4
    X[var] = db1[var].copy()

    # 'Price Sensitivity'
    var = 'Price Sensitivity'
    db1.loc[np.isnan(db1[var]), var] = 7
    X[var] = db1[var].copy()

    # 'Right Address'
    var = 'Right Address'
    db1.loc[pd.isnull(db1[var]), var] = 'Wrong'
    dummies = pd.get_dummies(db1[var], dummy_na=True)
    X = pd.concat([X, dummies], axis=1)

    # 'PhoneType'
    var = 'PhoneType'
    dummies = pd.get_dummies(db1[var], dummy_na=True)
    X = pd.concat([X, dummies], axis=1)

    # 'Premium Offered'
    var = 'Premium Offered'
    X[var] = db1[var].copy()

    # 'Estimated number of cars'
    var = 'Estimated number of cars'
    db1.loc[pd.isnull(db1[var]), var] = 0
    db1.loc[db1[var] == 'None', var] = 0
    db1.loc[db1[var] == 'One', var] = 1
    db1.loc[db1[var] == 'two', var] = 2
    db1.loc[db1[var] == 'Three', var] = 3
    X[var] = db1[var].copy()

    # 'Living Area (m^2)'
    var = 'Living Area (m^2)'
    db1.loc[pd.isnull(db1[var]), var] = 0
    X[var] = db1[var].copy()

    # 'Number of Fixed Lines'
    var = 'Number of Fixed Lines'
    db1.loc[pd.isnull(db1[var]), var] = 0
    X[var] = db1[var].copy()

    # 'yearBuilt'
    var = 'yearBuilt'
    db1.loc[pd.isnull(db1[var]), var] = db1[var].mean()
    X[var] = db1[var].copy()

    # 'Credit'
    var = 'Credit'
    db1.loc[pd.isnull(db1[var]), var] = 0
    X[var] = db1[var].copy()

    # 'Probability of Second Residence'
    var = 'Probability of Second Residence'
    db1.loc[pd.isnull(db1[var]), var] = 0
    db1.loc[db1[var] == 'Low', var] = 0
    db1.loc[db1[var] == 'Medium', var] = 1
    db1.loc[db1[var] == 'High', var] = 2
    X[var] = db1[var].copy()

    # Convert X to matrix
    Xmat = X.as_matrix()

    return Xmat



def neural_network(X_train=None, X_test=None, y_train=None, y_test=None, file=None):

    """Predict sales with a neural network

    Args:
        X_train (numpy.matrix) : X train
        X_test (numpy.matrix) : X train
        y_train (np.matrix): train y
        y_test (np.matrix): test y
        file (str) : path of a pickle file with data. If exists, overrules all the other inputs
    Returns:
        model:
    """

    if file is not None:
        # Load tha data
        data = pickle.load(open('ml_data.dat', 'rb'))
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=56))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae'])

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.15, verbose=1)

    # Test on the test set
    test_accuracy = model.test_on_batch(X_test, y_test, sample_weight=None)
    print('Test mean squared error: {:.3f}'.format(test_accuracy[0]))
    print('Test mean absolute arror: {:.3f}'.format(test_accuracy[1]))

    # Save model
    # serialize model to JSON
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('model.h5')
    print('Saved model to disk')

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    model = neural_network(file=r'ml_data.dat')
