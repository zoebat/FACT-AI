from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from fairGAN import Medgan
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import metrics
import pickle


def prepare_data(data, output_path):
    def label_fix(label):
        if label==' <=50K':
            return 0
        else:
            return 1
        
    data['income_bracket'] = data['income_bracket'].apply(label_fix)
    print("Base MI RACE;INCOME = ", mutual_info_classif(np.expand_dims(data['income_bracket'], -1), data['race'], discrete_features = True))
    print("Base MI Gender;INCOME = ", mutual_info_classif(np.expand_dims(data['income_bracket'], -1), data['gender'], discrete_features = True))
    
    for feat in ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country']:
        data[feat] = LabelEncoder().fit_transform(data[feat])

    data.drop(['capital_gain', 'capital_loss', 'education'], axis = 1, inplace = True)

    scaler = MinMaxScaler()
    scaler.fit(data)
    data[data.columns] = scaler.fit_transform(data)
    pickle.dump(data.values, open(output_path, "wb" ))


def train_fairgan(datapath):
    data = np.load(datapath, allow_pickle = True)
    inputDim = data.shape[1]-1
    inputNum = data.shape[0]
    tf.reset_default_graph()
    mg = Medgan(dataType='count',
                inputDim=inputDim,
                embeddingDim=128,
                randomDim=128,
                generatorDims=(128,128),
                discriminatorDims=(256,128),
                compressDims=(),
                decompressDims=(),
                bnDecay=0.99,
                l2scale=0.001)

    model_file = 'fair-999'
    out_file = 'fair'
    batch_size = 100
    
    mg.train(dataPath=datapath,
             modelPath=model_file,
             outPath=out_file,
             pretrainEpochs=200,
             nEpochs=1000,
             discriminatorTrainPeriod=2,
             generatorTrainPeriod=1,
             pretrainBatchSize=100,
             batchSize=batch_size,
             saveMaxKeep=0)

    return mg.generateData(nSamples=inputNum,
                        modelFile=model_file,
                        batchSize=batch_size,
                        outFile=out_file)

def generate_data(datapath, model_path, output_path, batch_size):
    data = np.load(datapath, allow_pickle = True)
    print("Original data shape", data.shape)
    inputDim = data.shape[1]
    inputNum = data.shape[0]
    tf.reset_default_graph()
    mg = Medgan(dataType='count',
                inputDim=inputDim,
                embeddingDim=128,
                randomDim=128,
                generatorDims=(128,128),
                discriminatorDims=(256,128),
                compressDims=(),
                decompressDims=(),
                bnDecay=0.99,
                l2scale=0.001)

   
    mg.generateData(nSamples=inputNum,
                        modelFile=model_path,
                        batchSize=batch_size,
                        outFile=output_path)

def fairgan_experiment(X_o, y_o , X_s, y_s):
    baseline_clf = MLPClassifier().fit(X_o, y_o)
    y_pred = baseline_clf.predict(X_o)

    print(
        "(baseline) scores: y vs y_pred",
        precision_score(y_o, y_pred),
        recall_score(y_o, y_pred),
        roc_auc_score(y_o, y_pred),
    )

    synth_clf = MLPClassifier().fit(X_s, y_s)
    y_pred_synth = synth_clf.predict(X_o)
    y_pred_synth_proba = synth_clf.predict_proba(X_o)

    print(
        "FTU",
        metrics.ftu(synth_clf, X_o, 9))
    print(
        "DP",
        metrics.dp(synth_clf, X_o, 9)
    )
    
    print(
        "scores: y vs y_pred_synth",
        precision_score(y_o, y_pred_synth),
        recall_score(y_o, y_pred_synth),
        roc_auc_score(y_o, y_pred_synth_proba[:, 1]))
    



if __name__ == "__main__":
    # data = pd.read_csv("census.csv")
    processed_data_path = 'preprocessed_data.npy';
    # #prepare_data(data, 'preprocessed_data.npy')

    model_path = 'fair-999'
    output_path = 'fair.npy'
    batch_size = 100
    generate_data(processed_data_path, model_path, output_path, batch_size)

    data_o = np.load(processed_data_path, allow_pickle = True)
    print(data_o.shape)
    X_o = data_o[:, 0:11]
    y_o = data_o[:, 11]

    data_s = np.load(output_path, allow_pickle = True)
    print(data_s.shape)
    X_s = data_s
    y_s = data_o[:X_s.shape[0], 11]
    
    fairgan_experiment(X_o, y_o , X_s, y_s)


