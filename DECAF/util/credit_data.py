import numpy as np
import pandas as pd
from sklearn import preprocessing

def load(bias):
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

    df.loc[df['ethnicity'] <= 4, 'ethnicity'] = 0
    df.loc[df['ethnicity'] > 4, 'ethnicity']= 1
    df.loc[df['ethnicity'] == 1 , 'employed'] =  1

    biased_data = df.copy()
    biased_data.loc[biased_data['ethnicity'] == 4, 'approved'] = np.logical_and(biased_data.loc[biased_data['ethnicity'] == 4, 'approved'].values, np.random.binomial(1, bias, len(biased_data.loc[biased_data['ethnicity'] == 4, 'approved']))).astype(int)

    print(biased_data.describe(percentiles=[.25, .5, .75, 0.90, 0.95, 0.99]))

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df)
    biased_data[biased_data.columns] = scaler.transform(biased_data)
    
    dfr = df.copy()
    biased_data = biased_data.values
    X = biased_data[:, :15].astype(np.float32)
    y = biased_data[:, 15].astype(np.uint32)
    Xy = biased_data
    return X, y, dfr, Xy, scaler