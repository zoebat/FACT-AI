from pyparsing import col
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from util import adult_data, metrics, data, dag
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from model.DECAF import DECAF

def experiment_train_base_classifier(X, y):
    baseline_clf = MLPClassifier().fit(X, y)
    y_pred = baseline_clf.predict(X)
    print("baseline scores", precision_score(y, y_pred), recall_score(y, y_pred), roc_auc_score(y, y_pred))

def experiment_decaf(X, y, Xy, min_max_scaler):
    dag_seed = [[9, 8], [0, 6], [9, 12], [11, 4], [0, 10], [0, 3], [5, 12], [8, 7], [6, 4], [8, 5], [5, 11], [0, 5], [10, 4], [8, 4], [2, 14], [1, 13], [5, 4], [9, 3], [9, 5], [3, 4], [0, 11], [0, 2], [7, 14], [6, 1], [0, 1], [13, 7], [9, 4], [1, 8], [7, 12], [12, 10], [11, 14], [9, 2], [6, 12], [11, 10], [12, 14], [4, 14], [1, 4], [5, 14], [8, 13], [8, 12], [1, 3], [7, 10], [0, 7], [0, 14], [2, 8], [3, 14], [6, 14], [1, 5], [11, 12], [10, 14], [6, 8], [9, 0], [13, 3], [5, 10], [2, 13], [5, 13], [9, 7], [13, 4], [9, 6], [9, 14], [12, 4], [5, 7]]
    bias_dict_FTU = {14: [9]}
    bias_dict_CF = {14: [9, 0, 4, 2, 5, 7]}
    bias_dict_DP = {14: [9, 0, 3, 4, 2, 12, 5, 6, 7]}

    baseline_clf = MLPClassifier().fit(X, y)
    y_pred = baseline_clf.predict(X)

    print(
        "(baseline) scores: y vs y_pred",
        precision_score(y, y_pred),
        recall_score(y, y_pred),
        roc_auc_score(y, y_pred),
    )
    dm = data.DataModule(Xy)
    model = DECAF(dm.dims[0], dag_seed=dag_seed, batch_size=64, lambda_gp=1, lambda_privacy=0, weight_decay=1e-2, grad_dag_loss=True, l1_W=1e-4, l1_g=0, use_mask=True)

    logger = TensorBoardLogger("logs", name="DECAF", log_graph=True)
    trainer = pl.Trainer(max_epochs=30, logger=logger)
    trainer.fit(model, dm)

    Xy_synth = ( model.gen_synthetic(dm.dataset.x, gen_order=model.get_gen_order(), biased_edges=bias_dict_DP).detach().numpy())
    Xy_synth1 = min_max_scaler.inverse_transform(Xy_synth)
    header = ['age','workclass','fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',  'relationship', 'race','sex', 'capital-gain', 'capital-loss',
      'hours-per-week',  'native-country', "label"]
    dfs  = pd.DataFrame(data=Xy_synth1, columns=header)
    print(dfs.describe(percentiles=[.25, .5, .75, 0.90, 0.95, 0.99]))

    X_synth = Xy_synth[:, :14]
    y_synth = np.round(Xy_synth[:, 14]).astype(int)

    y_base_synth = baseline_clf.predict(X_synth)
    synth_clf = MLPClassifier().fit(X_synth, y_synth)
    y_pred_synth = synth_clf.predict(X)
    y_pred_synth_proba = synth_clf.predict_proba(X)

    print(
        "FTU",
        metrics.ftu(synth_clf, X_synth, 9))
    print(
        "DP",
        metrics.dp(synth_clf, X_synth, 9)
    )
    
    print(
        "scores: y vs y_pred_synth",
        precision_score(y, y_pred_synth),
        recall_score(y, y_pred_synth),
        roc_auc_score(y, y_pred_synth_proba[:, 1]))
    

if __name__ == "__main__":
    X, y, dfr, Xy, min_max_scaler = adult_data.load()
    experiment_decaf(X, y, Xy, min_max_scaler)

