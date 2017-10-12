# -*- coding: utf-8 -*-
"""Predictive model"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Features to use
features = ['Number of Semesters Paid', 'ProdActive', 'ProdBought', 'Email', 'Province', 'Socieconomic Status',
            'TenureYears', 'NumberofCampaigns', 'Price Sensitivity', 'Right Address', 'PhoneType',
            'Estimated number of cars', 'Premium Offered', 'Living Area (m^2)', 'yearBuilt', 'Credit',
            'Number of Fixed Lines']

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

    # Matrix y
    y = db1.loc[:, 'Sales'].as_matrix()

    X = pd.DataFrame()

    # Number of Semesters Paid
    var = 'Number of Semesters Paid'
    db1.loc[np.isnan(db1[var]), var] = 0
    X[var] = db1[var].copy()

    # ProdActive
    var = 'ProdActive'
    X[var] = db1[var].copy()

    # Price Sensitivity
    var = 'Price Sensitivity'
    db1.loc[np.isnan(db1[var]), var] = 7
    X[var] = db1[var].copy()

    # ProdBought
    var = 'ProdBought'
    X[var] = db1[var].copy()

    # Email
    var = 'Email'
    X[var] = db1[var].copy()

    # Province
    var = 'Province'
    dummies = pd.get_dummies(db1[var], dummy_na=True)
    X = pd.concat([X, dummies], axis=1)

    # Socieconomic Status
    var = 'Socieconomic Status'
    db1.loc[pd.isnull(db1[var]), var] = 0
    db1.loc[db1[var] == 'Low', var] = 1
    db1.loc[db1[var] == 'Medium', var] = 2
    db1.loc[db1[var] == 'High', var] = 3
    db1.loc[db1[var] == 'Very High', var] = 4
    X[var] = db1[var].copy()

    # TenureYears
    var = 'TenureYears'
    db1[var] = 2013 - db1['Tenure']
    X[var] = db1[var].copy()

    # NumberofCampaigns
    var = 'NumberofCampaigns'
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

    # 'Estimated number of cars'
    var = 'Estimated number of cars'
    db1.loc[pd.isnull(db1[var]), var] = 0
    db1.loc[db1[var] == 'None', var] = 0
    db1.loc[db1[var] == 'One', var] = 1
    db1.loc[db1[var] == 'two', var] = 2
    db1.loc[db1[var] == 'Three', var] = 3
    X[var] = db1[var].copy()

    # 'Premium Offered'
    var = 'Premium Offered'
    X[var] = db1[var].copy()

    # 'Living Area (m^2)'
    var = 'Living Area (m^2)'
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

    # 'Number of Fixed Lines'
    var = 'Number of Fixed Lines'
    db1.loc[pd.isnull(db1[var]), var] = 0
    X[var] = db1[var].copy()

    # Convert X to matrix
    X = X.as_matrix()

    # Normalize X
    scaler = StandardScaler()
    Xnorm = scaler.fit_transform(X)

    # Train and test sets
    X_train, X_test, y_train, y_test = train_test_split(Xnorm, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    print(X_train)