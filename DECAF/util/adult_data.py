import numpy as np
import pandas as pd
from sklearn import preprocessing

def load():
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
    ]
    df = pd.read_csv(path, names=names, index_col=False)
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    for col in df:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]

    for feat in ['workclass', 'education','marital-status', 'occupation', 'relationship','race', 'sex', 'native-country', 'label']:
        df[feat] = preprocessing.LabelEncoder().fit_transform(df[feat])

    # print(df.describe(percentiles=[.25, .5, .75, 0.90, 0.95, 0.99]))

    dfr = df.copy()
    df = df.values

    min_max_scaler = preprocessing.MinMaxScaler()
    Xy = min_max_scaler.fit_transform(df)

    return dfr, Xy, min_max_scaler
