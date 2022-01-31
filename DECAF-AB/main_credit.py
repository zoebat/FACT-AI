from pyparsing import col
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from util import adult_data, credit_data, metrics
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from util import data
from model.DECAF import DECAF

def experiment_train_base_classifier(X, y):
    baseline_clf = XGBClassifier().fit(X, y)
    y_pred = baseline_clf.predict(X)
    print("baseline scores", precision_score(y, y_pred), recall_score(y, y_pred), roc_auc_score(y, y_pred))

def experiment_decaf(X, y, Xy, min_max_scaler):
    dag_seed = [[1, 7], [8, 1], [10, 2], [6, 1], [7, 10], [13, 2], [7, 2], [9, 10], [9, 15], [8, 7], [14, 15], [4, 9], [4, 3], [8, 15], [7, 11], [13, 11], [12, 9], [9, 8]]
    baseline_clf = MLPClassifier().fit(X, y)
    y_pred = baseline_clf.predict(X)

    print(
        "baseline scores",
        precision_score(y, y_pred),
        recall_score(y, y_pred),
        roc_auc_score(y, y_pred),
    )
    dm = data.DataModule(Xy)
    model = DECAF(dm.dims[0], dag_seed=dag_seed, batch_size=64, lambda_gp=1, lambda_privacy=0, weight_decay=1e-2, grad_dag_loss=True, l1_W=1e-4, l1_g=0, use_mask=True)

    logger = TensorBoardLogger("logs", name="DECAF", log_graph=True)
    trainer = pl.Trainer(max_epochs=50, logger=logger)
    trainer.fit(model, dm)

    Xy_synth = ( model.gen_synthetic(dm.dataset.x, gen_order=model.get_gen_order()).detach().numpy())
    Xy_synth = min_max_scaler.inverse_transform(Xy_synth)
    # header = ['age','workclass','fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',  'relationship', 'race','sex', 'capital-gain', 'capital-loss',
    #   'hours-per-week',  'native-country']
    dfs  = pd.DataFrame(data=Xy_synth)
    print(dfs.describe(percentiles=[.25, .5, .75, 0.90, 0.95, 0.99]))

    X_synth = Xy_synth[:, :15]
    y_synth = np.round(Xy_synth[:, 15])
    print(max(y_synth), min(y_synth))

    synth_clf = MLPClassifier().fit(X_synth, y_synth)
    y_pred_synth = synth_clf.predict(X_synth)
    print(
        "FTU",
        metrics.ftu(synth_clf, X_synth, 6))
    print(
        "DP",
        metrics.dp(synth_clf, X_synth, 6)
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
    X, y, dfr, Xy, min_max_scaler = credit_data.load()
    experiment_decaf(X, y, Xy, min_max_scaler)