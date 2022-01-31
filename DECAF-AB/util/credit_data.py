import numpy as np
import pandas as pd
from sklearn import preprocessing

def load():
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    names = [
        'male', 
        'age', 
        'debt', 
        'married', 
        'bankcustomer', 
        'educationlevel', 
        'ethnicity', 
        'yearsemployed',
        'priordefault', 
        'employed', 
        'creditscore', 
        'driverslicense', 
        'citizen', 
        'zip', 
        'income', 
        'approved'
        ]

    df = pd.read_csv(path, names=names, index_col=False)
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    df.reset_index(drop=True, inplace=True) 

    for col in df:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]

    df = df.dropna(how = 'all')
    df = df[df.age != '?']

    for feat in ['male', 'married','bankcustomer', 'educationlevel', 'ethnicity','priordefault', 'employed', 'driverslicense', 'citizen', 'zip', 'approved']:
        df[feat] = preprocessing.LabelEncoder().fit_transform(df[feat])

    print(df.describe(percentiles=[.25, .5, .75, 0.90, 0.95, 0.99]))
    
    dfr = df.copy()
    df = df.values
    X = df[:, :15].astype(np.float32)
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    y = df[:, 15].astype(np.uint32)
    Xy = min_max_scaler.fit_transform(df)
    return X, y, dfr, Xy, min_max_scaler