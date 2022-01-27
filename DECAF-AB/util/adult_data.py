from cProfile import label
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler, MinMaxScaler, RobustScaler

def load():
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
    discrete_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'label']
    label =  ['label']
    df = pd.read_csv(path, names=columns, index_col=False)
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    df_discrete = df[list(set(discrete_columns) - set(label))]
    df_continuous = df[list(set(columns) - set(discrete_columns))]
    df_label = df['label']

    print(df.describe())

    continuous_columns = df_continuous.columns.tolist()
    discrete_columns = df_discrete.columns.tolist()

    discrete_data = df_discrete.values
    continuous_data = df_continuous.values


    encoder = OrdinalEncoder()
    discrete_data = encoder.fit_transform(discrete_data)

    
    data = np.concatenate((continuous_data ,discrete_data),axis=1)

    df = pd.DataFrame(data=data, columns= continuous_columns + discrete_columns)

    print(df.describe())

    y = df_label.apply(lambda x: 0 if x == '<=50K' else 1)
    y = y.values.astype(np.uint8)

    X = df.values
    Xy = np.concatenate((X , np.expand_dims(y, axis=1)), axis=1)

    print(Xy[0:20])

    return X, y, Xy

if __name__ == "__main__":
    load()