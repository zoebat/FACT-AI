import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from util import adult_data
from xgboost import XGBClassifier

from util import data
from model.DECAF import CasualGAN

def experiment_train_base_classifier(X, y):
    baseline_clf = XGBClassifier().fit(X, y)
    y_pred = baseline_clf.predict(X)
    print("baseline scores", precision_score(y, y_pred), recall_score(y, y_pred), roc_auc_score(y, y_pred))

def experiment_decaf(X, y):
    dag_seed = [
        [0, 6],
        [0, 12],
        [0, 5],
        [0, 1],
        [0, 3],
        [0, 7],
        [8, 6],
        [8, 12],
        [8, 3],
        [8, 5],
        [9, 6],
        [9, 5],
        [9,12],
        [9,1],
        [9,4],
        [9, 7],
        [13, 5],
        [13, 12],
        [13, 4],
        [13, 1],
        [13, 7],
        [5, 6],
        [5, 12],
        [5, 1],
        [5, 7],
        [5, 3],
        [4, 6],
        [4, 12],
        [4, 1],
        [4, 7]
        ]

    dm = data.DataModule(X)
    model = CasualGAN(dm.dims[0], dag_seed=dag_seed, batch_size=64, lambda_gp=1, lambda_privacy=0, weight_decay=1e-2, grad_dag_loss=True, l1_W=1e-4, l1_g=0, p_gen=-1, use_mask=True)

    trainer = pl.Trainer(max_epochs=1, logger=False)
    trainer.fit(model, dm)

    X_synth = ( model.gen_synthetic(dm.dataset.x, gen_order=model.get_gen_order()).detach().numpy())
    print(X_synth[0:5])

if __name__ == "__main__":
    X, y, Xy = adult_data.load()
    print(X[0:5,])
    print(y[0:5,])
    experiment_decaf(X, y)