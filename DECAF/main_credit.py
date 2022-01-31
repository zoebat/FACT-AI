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
from sklearn.model_selection import train_test_split

from util import data
from model.DECAF import DECAF

def experiment_train_base_classifier(X, y):
    baseline_clf = MLPClassifier().fit(X, y)
    y_pred = baseline_clf.predict(X)
    print("baseline scores", precision_score(y, y_pred), recall_score(y, y_pred), roc_auc_score(y, y_pred))

def experiment_decaf(X, y, Xy, min_max_scaler):
    dag_seed = [[1, 7], [7, 10], [10, 2], [10, 15], [2, 14], [14, 15], [14, 3], [3, 15], [13, 3], [12, 3], [6, 3], [6, 15], [5, 3], [5, 9], [9, 2], [9, 15], [9, 4], [9, 12], [11, 9], [8, 10], [8, 15], [8, 9]]
    baseline_clf = MLPClassifier().fit(X, y)
    y_pred = baseline_clf.predict(X)
    bias_dict_FT = {15: [6]}
    bias_dict_DP = {15: [6, 3]}
    X_train, X_test, y_train, y_test = train_test_split(Xy[:, :15], Xy[:, 15], test_size=0.10, random_state=42)

    print(
        "baseline scores",
        precision_score(y, y_pred),
        recall_score(y, y_pred),
        roc_auc_score(y, y_pred),
    )
    dm = data.DataModule(np.c_[X_train, y_train])
    # dm = data.DataModule(Xy)
    model = DECAF(dm.dims[0], dag_seed=dag_seed, h_dim=100, batch_size=32, lambda_gp=1, lambda_privacy=0, weight_decay=1e-2, grad_dag_loss=True, l1_W=1e-4, l1_g=0, use_mask=True)

    logger = TensorBoardLogger("logs", name="DECAF", log_graph=True)
    trainer = pl.Trainer(max_epochs=250, logger=logger)
    trainer.fit(model, dm)

    Xy_synth = ( model.gen_synthetic(dm.dataset.x, gen_order=model.get_gen_order(), biased_edges={}).detach().numpy())
    Xy_synth = min_max_scaler.inverse_transform(Xy_synth)
    header = [
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
    dfs  = pd.DataFrame(data=Xy_synth, columns=header)
    print("columns:", len(dfs.columns))
    print(dfs.describe(percentiles=[.25, .5, .75, 0.90, 0.95, 0.99]))

    X_synth = Xy_synth[:, :15]
    y_synth = np.round(Xy_synth[:, 15])
    print(max(y_synth), min(y_synth))

    y_base_synth = baseline_clf.predict(X_synth)
    synth_clf = MLPClassifier().fit(X_synth, y_synth)
    y_pred_synth = synth_clf.predict(X_test)
    print(min(y_pred_synth), max(y_pred_synth))
    y_pred_synth_proba = synth_clf.predict_proba(X_test)

    print(
        "FTU",
        metrics.ftu(synth_clf, X_synth, 6))
    print(
        "DP",
        metrics.dp(synth_clf, X_synth, 6)
    )
    
    print(
        "scores: y vs y_pred_synth",
        precision_score(y_test, y_pred_synth),
        recall_score(y_test, y_pred_synth),
        roc_auc_score(y_test, y_pred_synth_proba[:, 1]))
    
    return [precision_score(y_test, y_pred_synth), recall_score(y_test, y_pred_synth), roc_auc_score(y_test, y_pred_synth_proba[:, 1]), metrics.ftu(synth_clf, X_synth, 6), metrics.dp(synth_clf, X_synth, 6)]

if __name__ == "__main__":
    n = 6
    betas = np.linspace(0, 1, n, endpoint=True)
    results_DP = []
    for beta in betas:
        X, y, dfr, Xy, min_max_scaler = credit_data.load(beta)
        list_results = experiment_decaf(X, y, Xy, min_max_scaler)
        results_DP.append(list_results)
    results_DP = np.array(results_DP)
    np.save("results_ND2.npy", results_DP)
    # X, y, dfr, Xy, min_max_scaler = credit_data.load(1)
    # experiment_decaf(X, y, Xy, min_max_scaler)
