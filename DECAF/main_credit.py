from pyparsing import col
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from util import adult_data, credit_data
from xgboost import XGBClassifier

from util import data
from model.DECAF import DECAF

def experiment_train_base_classifier(X, y):
    baseline_clf = XGBClassifier().fit(X, y)
    y_pred = baseline_clf.predict(X)
    print("baseline scores", precision_score(y, y_pred), recall_score(y, y_pred), roc_auc_score(y, y_pred))

def experiment_decaf(X, y, min_max_scaler):
    dag_seed = [[0, 6], [7, 0], [9, 1], [5, 0], [6, 9], [12, 1], [6, 1], [8, 9], [8, 14], [7, 6], [13, 14], [3, 8], [3, 2], [7, 14], [6, 10], [12, 10], [11, 8], [8, 7]]
    baseline_clf = XGBClassifier().fit(X, y)
    y_pred = baseline_clf.predict(X)

    print(
        "baseline scores",
        precision_score(y, y_pred),
        recall_score(y, y_pred),
        roc_auc_score(y, y_pred),
    )
    dm = data.DataModule(X)
    model = DECAF(dm.dims[0], dag_seed=dag_seed, batch_size=64, lambda_gp=1, lambda_privacy=0, weight_decay=1e-2, grad_dag_loss=True, l1_W=1e-4, l1_g=0, use_mask=True)

    logger = TensorBoardLogger("logs", name="DECAF", log_graph=True)
    trainer = pl.Trainer(max_epochs=30, logger=logger)
    trainer.fit(model, dm)

    X_synth = ( model.gen_synthetic(dm.dataset.x, gen_order=model.get_gen_order()).detach().numpy())
    X_synth = min_max_scaler.inverse_transform(X_synth)
    # header = ['age','workclass','fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',  'relationship', 'race','sex', 'capital-gain', 'capital-loss',
    #   'hours-per-week',  'native-country']
    dfs  = pd.DataFrame(data=X_synth)
    print(dfs.describe(percentiles=[.25, .5, .75, 0.90, 0.95, 0.99]))

    y_synth = baseline_clf.predict(X_synth)
    print(y_synth[0:20])

    synth_clf = XGBClassifier().fit(X_synth, y_synth)
    y_pred_synth = synth_clf.predict(X_synth)
    print(y_pred[0:20])
    print(
        "synth scores",
        precision_score(y, y_pred_synth),
        recall_score(y, y_pred_synth),
        roc_auc_score(y, y_pred_synth),
    )

if __name__ == "__main__":
    X, y, dfr, min_max_scaler = credit_data.load()
    experiment_decaf(X, y, min_max_scaler)