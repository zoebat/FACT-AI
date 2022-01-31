from pyparsing import col
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from util import adult_data
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from util import data, metrics
from model.DECAF import DECAF

def experiment_train_base_classifier(X, y):
    baseline_clf = XGBClassifier().fit(X, y)
    y_pred = baseline_clf.predict(X)
    print("baseline scores", precision_score(y, y_pred), recall_score(y, y_pred), roc_auc_score(y, y_pred))

def experiment_decaf(X, y, Xy):
    dag_seed = [
        [0, 6],
        [0, 12],
        [0, 14],
        [0, 1],
        [0, 5],
        [0, 3],
        [1, 14],
        [3, 6],
        [3, 12],
        [3, 14],
        [3, 1],
        [3, 7],
        [5, 6],
        [5, 12],
        [5, 14],
        [5, 1],
        [5, 7],
        [5, 3],
        [6, 14],
        [7, 14],
        [8, 6],
        [8, 14],
        [8, 12],
        [8, 3],
        [8, 5],
        [9, 6],
        [9, 5],
        [9, 14],
        [9, 12],
        [9, 1],
        [9, 3],
        [9, 7],
        [12, 14],
        [13, 5],
        [13, 12],
        [13, 3],
        [13, 1],
        [13, 14],
        [13, 7],
    ]
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
    trainer = pl.Trainer(max_epochs=20, logger=False)
    trainer.fit(model, dm)

    Xy_synth = ( model.gen_synthetic(dm.dataset.x, gen_order=model.get_gen_order()).detach().numpy())
    # Xy_synth = min_max_scaler.inverse_transform(Xy_synth)
    # header = ['age','workclass','fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',  'relationship', 'race','sex', 'capital-gain', 'capital-loss',
    #   'hours-per-week',  'native-country', "label"]
    # dfs  = pd.DataFrame(data=Xy_synth, columns=header)
    # print(dfs.describe(percentiles=[.25, .5, .75, 0.90, 0.95, 0.99]))

    X_synth = Xy_synth[:, :14]
    y_synth = np.round(Xy_synth[:, 14])

    synth_clf = MLPClassifier().fit(X_synth, y_synth)
    y_pred_synth = synth_clf.predict(X_synth)

    print(
        "FTU",
        metrics.ftu(synth_clf, X_synth, 9))
    print(
        "DP",
        metrics.dp(synth_clf, X_synth, 9)
    )

    print(
        "scores: y_pred vs y_pred_synth",
        precision_score(y_pred, y_pred_synth),
        recall_score(y_pred, y_pred_synth),
        roc_auc_score(y_pred, y_pred_synth),
    )

    print(
        "scores: y vs y_pred_synth",
        precision_score(y, y_pred_synth),
        recall_score(y, y_pred_synth),
        roc_auc_score(y, y_pred_synth),
    )

    print(
        "scores: y_pred vs y_synth",
        precision_score(y_pred, y_synth),
        recall_score(y_pred, y_synth),
        roc_auc_score(y_pred, y_synth),
    )

    print(
        "scores: y vs y_synth",
        precision_score(y, y_synth),
        recall_score(y, y_synth),
        roc_auc_score(y, y_synth),
    )

if __name__ == "__main__":
    X, y, Xy = adult_data.load()
    experiment_decaf(X, y, Xy)
