from pyparsing import col
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from util import crime_data, metrics, data, dag
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from model.DECAF import DECAF

def experiment_decaf(X, y, Xy, min_max_scaler):
    dag_seed = dag.find_dag('crime')
    # print(dag_seed)

    baseline_clf = XGBClassifier().fit(X,y)
    y_pred_base = baseline_clf.predict(X)

    print(
        "baseline scores",
        precision_score(y, y_pred_base),
        recall_score(y, y_pred_base),
        roc_auc_score(y, y_pred_base),
    )
    dm = data.DataModule(Xy)
    model = DECAF(dm.dims[0], dag_seed=dag_seed, batch_size=32, lambda_gp=1, lambda_privacy=0, weight_decay=1e-2, grad_dag_loss=True, l1_W=1e-4, l1_g=0, use_mask=True)

    logger = TensorBoardLogger("logs", name="DECAF", log_graph=True)
    trainer = pl.Trainer(max_epochs=150, logger=logger)
    trainer.fit(model, dm)

    Xy_synth = ( model.gen_synthetic(dm.dataset.x, gen_order=model.get_gen_order(), biased_edges={}).detach().numpy())
    Xy_synth = min_max_scaler.inverse_transform(Xy_synth)
    dfs  = pd.DataFrame(data=Xy_synth)
    print(dfs.describe(percentiles=[.25, .5, .75, 0.90, 0.95, 0.99]))

    X_synth = Xy_synth[:, :127]
    y_synth = Xy_synth[:, 127]

    y_median = np.median(y_synth)
    y_synth = np.where(y_synth > y_median, 1, 0)

    print(y_synth)
    print(max(y_synth), min(y_synth))

    synth_clf = XGBClassifier().fit(X_synth, y_synth)
    y_pred = synth_clf.predict(X)
    # print(max(y_pred_synth), min(y_pred_synth))
    print(
        "FTU",
        metrics.ftu(synth_clf, X_synth, 7))
    print(
        "DP",
        metrics.dp(synth_clf, X_synth, 7)
    )

    print(
        "scores: y_pred_base vs y_pred",
        precision_score(y_pred_base, y_pred),
        recall_score(y_pred_base, y_pred),
        roc_auc_score(y_pred_base, y_pred),
    )

    print(
        "scores: y vs y_pred",
        precision_score(y, y_pred),
        recall_score(y, y_pred),
        roc_auc_score(y, y_pred)
    )

if __name__ == "__main__":
    X, y, dfr, Xy, min_max_scaler = crime_data.load()
    print(dfr['ViolentCrimesPerPop'])
    experiment_decaf(X, y, Xy, min_max_scaler)