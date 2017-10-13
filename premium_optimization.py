# -*- coding: utf-8 -*-
"""Premium optimization"""

import pandas as pd
from keras.models import model_from_json
from predictive_model import proccess_X

def predict(premium, db2=None):
    """Predicts sales for a specified output

    Args:
        premium (float) : premium offered
        db2 (pandas.DataFrame) : data
            If it's not provided, the function will read from 'Database.xlsx'
    Returns:
        expected_income (float) : Expected income per customer
        expected_sales (float) : Expected sales probability
        y (np.array) : Array with individual sales (0, 1) for each customer
    """

    if db2 is None:
        # Read data
        xls = pd.ExcelFile('Database.xlsx')
        db2 = xls.parse(2)

    # Fill the premium column
    db2['Premium Offered'] = premium

    # Generate an X matrix
    X = proccess_X(db2)

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
    expected_sales = y.mean()
    expected_income = premium * expected_sales

    return expected_income, expected_sales, y


if __name__ == "__main__":
    #premium = 12
    #expected_income, expected_sales, y = predict(premium)
    #print(expected_income, expected_sales)
    xls = pd.ExcelFile('Database.xlsx')
    db2 = xls.parse(2)
    print(db2.columns)








