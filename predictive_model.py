# -*- coding: utf-8 -*-
"""Predictive model"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib


# Features to use
features = ['ProdActive', 'ProdBought', 'NumberofCampaigns', 'Email', 'Province', 'Tenure', 'Socieconomic Status',
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
    db2 = xls.parse(2)

    db1.loc[np.isnan(db1['Number of Semesters Paid']), 'Number of Semesters Paid'] = 0
    y = db1.loc[:, 'Number of Semesters Paid'].as_matrix()

    # Fill the premium column in db2
    db2['Premium Offered'] = db1['Premium Offered'].mean()

    # To get all columns in X, we need to mix it with the training data
    db3 = pd.concat([db1[features], db2[features]], axis=0)

    # Generate an X matrix
    Xall = proccess_X(db3)
    X = Xall[:db1.shape[0], :]
    X2 = Xall[db1.shape[0]:, :]

    # Train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Pickle the data
    data = {'X1': X, 'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'X2': X2}
    pickle.dump(data, open('ml_data.dat', 'wb'))

    return X_train, X_test, y_train, y_test


def proccess_X(db):
    """Create X matrix (common for train and test data)

    Args:
        db (pandas.DataFrame)

    Returns:
        Xmat (np.matrix): X
    """

    db1 = db.copy()

    X = pd.DataFrame()

    # 'Premium Offered'
    var = 'Premium Offered'
    X[var] = db1[var].copy()

    # 'Price Sensitivity'
    var = 'Price Sensitivity'
    db1.loc[np.isnan(db1[var]), var] = 13
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
    db1.loc[pd.isnull(db1[var]), var] = 'Unknown'
    le = LabelEncoder()
    db1[var] = le.fit_transform(db1[var]).reshape(-1, 1)
    X[var] = db1[var].copy()

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

    # 'Right Address'
    var = 'Right Address'
    db1.loc[pd.isnull(db1[var]), var] = 'Wrong'
    #dummies = pd.get_dummies(db1[var], dummy_na=True, prefix=var)
    le = LabelEncoder()
    db1[var] = le.fit_transform(db1[var]).reshape(-1, 1)
    X[var] = db1[var].copy()

    # 'PhoneType'
    var = 'PhoneType'
    #dummies = pd.get_dummies(db1[var], dummy_na=True, prefix=var)
    le = LabelEncoder()
    db1[var] = le.fit_transform(db1[var]).reshape(-1, 1)
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


def random_forest(X_train=None, X_test=None, y_train=None, y_test=None, file=None):

    """Predict sales with a random forest

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

    model = RandomForestRegressor(n_estimators=100, max_features=0.5, n_jobs=4, verbose=1)

    print("Training...")
    # Your model is trained on the training_data
    model.fit(X_train, y_train)

    # Test on the test set
    y_pred = model.predict(X_test)
    accuracy = np.mean((y_pred - y_test)**2)
    print('Test mean squared error: {:.3f}'.format(accuracy))

    # Save model
    joblib.dump(model, 'random_forest.pkl')

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    model = random_forest(file=r'ml_data.dat')
