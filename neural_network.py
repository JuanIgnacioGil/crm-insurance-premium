# -*- coding: utf-8 -*-
"""Predictive model"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LeakyReLU
from keras.regularizers import l1, l2

from random_forest import selected_features

# Features to use
features = ['ProdActive', 'ProdBought', 'NumberofCampaigns', 'Email', 'Province', 'Tenure', 'Socieconomic Status',
            'Price Sensitivity', 'Right Address', 'PhoneType','Premium Offered', 'Estimated number of cars',
            'Living Area (m^2)', 'Number of Fixed Lines', 'yearBuilt', 'Credit', 'Probability of Second Residence']

def prepare_data(features=None):
    """Prepare data for analysis

    Args:
        features (list of str): list with features

    Returns:
        X_train (np.matrix): train X
        X_test (np.matrix): test X
        y_train (np.matrix): train y
        y_test (np.matrix): test y
        scaler (sklearn.preprocessing.StandardScaler) : scaler
    """

    # Read data
    xls = pd.ExcelFile('Database.xlsx')
    db1 = xls.parse(1)
    db2 = xls.parse(2)

    db1.loc[np.isnan(db1['Number of Semesters Paid']), 'Number of Semesters Paid'] = 0
    y = (db1['Number of Semesters Paid'] * db1['Premium Offered']).as_matrix()

    # Fill the premium column in db2
    db2['Premium Offered'] = db1['Premium Offered'].mean()

    # To get all columns in X, we need to mix it with the training data
    if features is None:
        features = [x for x in db1.columns if x not in not_features]

    db3 = pd.concat([db1[features], db2[features]], axis=0)

    # Generate an X matrix
    Xall = proccess_X(db3, features)

    X = Xall[:db1.shape[0], :]
    X2 = Xall[db1.shape[0]:, :]

    # Normalize X
    scaler = StandardScaler()
    Xnorm = scaler.fit_transform(X)

    # Train and test sets
    X_train, X_test, y_train, y_test = train_test_split(Xnorm, y, test_size=0.15, random_state=42)

    # Pickle the data
    data = {'X1': X, 'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'scaler': scaler, 'X2': X2}
    pickle.dump(data, open('nn_data.dat', 'wb'))

    return X_train, X_test, y_train, y_test, scaler


def proccess_X(db, features):
    """Create X matrix (common for train and test data)

    Args:
        db (pandas.DataFrame)
        features (list of str)

    Returns:
        Xmat (np.matrix): X
    """

    db1 = db.copy()

    X = pd.DataFrame()

    # 'Premium Offered'
    var = 'Premium Offered'
    if var in features:
        X[var] = db1[var].copy()

    # 'Price Sensitivity'
    var = 'Price Sensitivity'
    if var in features:
        db1.loc[np.isnan(db1[var]), var] = 7
        X[var] = db1[var].copy()

    # 'PhoneType'
    var = 'PhoneType'
    if var in features:
        dummies = pd.get_dummies(db1[var], dummy_na=True, prefix=var)
        X = pd.concat([X, dummies], axis=1)

    # Email
    var = 'Email'
    if var in features:
        X[var] = db1[var].copy()

    # Tenure
    var = 'Tenure'
    if var in features:
        X[var] = db1[var].copy()

    # NumberofCampaigns
    var = 'NumberofCampaigns'
    if var in features:
        X[var] = db1[var].copy()

    # ProdActive
    var = 'ProdActive'
    if var in features:
        X[var] = db1[var].copy()

    # ProdBought
    var = 'ProdBought'
    if var in features:
        X[var] = db1[var].copy()

    # 'Birthdate'
    var = 'Birthdate'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    # Socieconomic Status
    var = 'Socieconomic Status'
    if var in features:
        db1.loc[db1[var] == 'Low', var] = 1
        db1.loc[db1[var] == 'Medium', var] = 2
        db1.loc[db1[var] == 'High', var] = 3
        db1.loc[db1[var] == 'Very High', var] = 4
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    # Province
    var = 'Province'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = 'Unknown'
        dummies = pd.get_dummies(db1[var], dummy_na=True, prefix=var)
        X = pd.concat([X, dummies], axis=1)

    # 'Right Address'
    var = 'Right Address'
    if var in features:
        db1.loc[db1[var] == 'Right', var] = 1
        db1.loc[db1[var] == 'Wrong', var] = 0
        db1.loc[pd.isnull(db1[var]), var] = db1[var].mean()
        X[var] = db1[var].copy()

    # 'Living Area (m^2)'
    var = 'Living Area (m^2)'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    # 'House Price'
    var = 'House Price'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    # 'Income'
    var = 'Income'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    # 'yearBuilt'
    var = 'yearBuilt'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = db1[var].mean()
        X[var] = db1[var].copy()

    # 'House Insurance'
    var = 'House Insurance'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    # 'Pension Plan'
    var = 'Pension Plan'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    # 'Estimated number of cars'
    var = 'Estimated number of cars'
    if var in features:
        db1.loc[db1[var] == 'None', var] = 0
        db1.loc[db1[var] == 'One', var] = 1
        db1.loc[db1[var] == 'two', var] = 2
        db1.loc[db1[var] == 'Three', var] = 3
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    # 'Probability of Second Residence'
    var = 'Probability of Second Residence'
    if var in features:
        db1.loc[db1[var] == 'Low', var] = 0.25
        db1.loc[db1[var] == 'Medium', var] = 0.5
        db1.loc[db1[var] == 'High', var] = 0.75
        db1.loc[pd.isnull(db1[var]), var] = db1[var].mean()
        X[var] = db1[var].copy()

    # 'Credit'
    var = 'Credit'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    # 'Savings'
    var = 'Savings'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    # 'Number of Mobile Phones'
    var = 'Number of Mobile Phones'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    # 'Number of Fixed Lines'
    var = 'Number of Fixed Lines'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    # ADSL
    var = 'ADSL'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    # '3G Devices'
    var = '3G Devices'
    if var in features:
        db1.loc[pd.isnull(db1[var]), var] = db1[var].median()
        X[var] = db1[var].copy()

    ## 'Type of House'
    var = 'Type of House'
    if var in features:
        dummies = pd.get_dummies(db1[var], dummy_na=True, prefix=var)
        X = pd.concat([X, dummies], axis=1)

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
        data = pickle.load(open('nn_data.dat', 'rb'))
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=54))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='relu'))
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
    with open('nn_model.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('nn_model.h5')
    print('Saved model to disk')

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = prepare_data(selected_features)
    model = neural_network(file=r'nn_data.dat')