# -*- coding: utf-8 -*-
"""Premium optimization"""

import pandas as pd
import pickle
from keras.models import model_from_json
from predictive_model import proccess_X, features


def predict(premium, db2=None, db1=None):
    """Predicts sales for a specified output

    Args:
        premium (float) : premium offered
        db1 (pandas.DataFrame) : train data
            If it's not provided, the function will read from 'Database.xlsx'
        db2 (pandas.DataFrame) : data
            If it's not provided, the function will read from 'Database.xlsx'
    Returns:
        expected_income (float) : Expected income per customer
        expected_semesters_paid (float) : Expected number of semesters paid
        expected_sales (float) : Expected sales probability
        y (np.array) : Array with individual sales (0, 1) for each customer
    """

    if (db2 is None) | (db1 is None):
        # Read data
        xls = pd.ExcelFile('Database.xlsx')
        db1 = xls.parse(1)
        db2 = xls.parse(2)

    # Fill the premium column
    db2['Premium Offered'] = premium

    # To get all columns in X, we need to mix it with the training data
    db3 = pd.concat([db2[features], db1[features]], axis=0)

    # Generate an X matrix
    Xall = proccess_X(db3)
    X = Xall[:db2.shape[0], :]

    # Read the model
    data = pickle.load(open('ml_data.dat', 'rb'))
    scaler = data['scaler']
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('model.h5')

    # Predict X data
    Xnorm = scaler.transform(X)
    y = model.predict(Xnorm)
    expected_semesters_paid = y.mean()
    expected_sales = y[y > 0.5].count() / y.count()
    expected_income = premium * expected_semesters_paid

    return expected_income, expected_semesters_paid, expected_sales, y


if __name__ == "__main__":
    premium = 12
    expected_income, expected_sales, y = predict(premium)
    print(expected_income, expected_sales)








