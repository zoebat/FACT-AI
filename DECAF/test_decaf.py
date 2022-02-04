from re import L
import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from util import adult_data, credit_data, crime_data, metrics, data, dag
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything

from models.DECAF import DECAF

def experiment_decaf(dataset_name, dag_seed, bias_dict, log_dir, bias=0):
    if dataset_name == 'adult':
        _, Xy, min_max_scaler = adult_data.load()
        logger = TensorBoardLogger("logs", name=log_dir, log_graph=True)
        trainer = pl.Trainer(max_epochs=50, logger=logger)
        protected_attribute = 9
    elif dataset_name == 'credit':
        _, Xy, min_max_scaler = credit_data.load(bias)
        logger = TensorBoardLogger("logs", name=log_dir, log_graph=True)
        trainer = pl.Trainer(max_epochs=250, logger=logger)
        protected_attribute = 6
    elif dataset_name == 'crime':
        _, Xy, min_max_scaler = crime_data.load()
        logger = TensorBoardLogger("logs", name=log_dir, log_graph=True)
        trainer = pl.Trainer(max_epochs=150, logger=logger)
        protected_attribute = 7
    else:
        raise ValueError("use one of the following three: 'adult', 'credit', 'crime'")

    X_train, X_test, y_train, y_test = train_test_split(Xy[:, :-1], Xy[:, -1], test_size=0.10, random_state=42)
    baseline_clf = MLPClassifier().fit(X_train, y_train)
    y_pred = baseline_clf.predict(X_test)

    print(
        "baseline scores",
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        roc_auc_score(y_test, y_pred),
    )

    dm = data.DataModule(np.c_[X_train, y_train])
    model = DECAF(dm.dims[0], dag_seed=dag_seed, h_dim=100, batch_size=32, lambda_gp=1, lambda_privacy=0, weight_decay=1e-2, grad_dag_loss=True, l1_W=1e-4, l1_g=0, use_mask=True)

    trainer.fit(model, dm)

    Xy_synth = (model.gen_synthetic(dm.dataset.x, gen_order=model.get_gen_order(), biased_edges=bias_dict).detach().numpy())
    Xy_synth = min_max_scaler.inverse_transform(Xy_synth)

    X_synth = Xy_synth[:, :-1]
    y_synth = np.round(Xy_synth[:, -1])

    synth_clf = MLPClassifier().fit(X_synth, y_synth)
    y_pred_synth = synth_clf.predict(X_test)
    y_pred_synth_proba = synth_clf.predict_proba(X_test)

    print(
        "FTU",
        metrics.ftu(synth_clf, X_synth, protected_attribute))
    print(
        "DP",
        metrics.dp(synth_clf, X_synth, protected_attribute))
    
    print(
        "scores: y_test vs y_pred_synth",
        precision_score(y_test, y_pred_synth),
        recall_score(y_test, y_pred_synth),
        roc_auc_score(y_test, y_pred_synth_proba[:, 1]))


if __name__ == "__main__":
    seeds = list(range(10))
    n = 6
    betas = np.linspace(0, 1, n, endpoint=True)
    adult_dag = [[0, 6], [0, 12], [0, 14], [0, 1], [0, 5], [0, 3], [1, 14], [3, 6], [3, 12], [3, 14], [3, 1], 
                 [3, 7], [5, 6], [5, 12], [5, 14], [5, 1], [5, 7], [5, 3], [6, 14], [7, 14], [8, 6], [8, 14], 
                 [8, 12], [8, 3], [8, 5], [9, 6], [9, 5], [9, 14], [9, 12], [9, 1], [9, 3], [9, 7], [12, 14], 
                 [13, 5], [13, 12], [13, 3], [13, 1], [13, 14], [13, 7]]
    credit_dag = [[1, 7], [7, 10], [10, 2], [10, 15], [2, 14], [14, 15], [14, 3], [3, 15], [13, 3], [12, 3], 
                  [6, 3], [6, 15], [5, 3], [5, 9], [9, 2], [9, 15], [9, 4], [9, 12], [11, 9], [8, 10], [8, 15], 
                  [8, 9]]
    adult_dag2 = [[9, 8], [0, 6], [9, 12], [11, 4], [0, 10], [0, 3], [5, 12], [8, 7], [6, 4], [8, 5], [5, 11],
                  [0, 5], [10, 4], [8, 4], [2, 14], [1, 13], [5, 4], [9, 3], [9, 5], [3, 4], [0, 11], [0, 2], 
                  [7, 14], [6, 1], [0, 1], [13, 7], [9, 4], [1, 8], [7, 12], [12, 10], [11, 14], [9, 2], [6, 12], 
                  [11, 10], [12, 14], [4, 14], [1, 4], [5, 14], [8, 13], [8, 12], [1, 3], [7, 10], [0, 7], [0, 14], 
                  [2, 8], [3, 14], [6, 14], [1, 5], [11, 12], [10, 14], [6, 8], [9, 0], [13, 3], [5, 10], [2, 13], 
                  [5, 13], [9, 7], [13, 4], [9, 6], [9, 14], [12, 4], [5, 7]]
    adult_FTU = {14: [9]}
    adult_CF = {14: [7, 9, 5]}
    adult_DP = {14: [7, 13, 1, 9, 5, 12, 6]}
    adult_dag2_FTU = {14: [9]}
    adult_dag2_CF = {14: [9, 0, 4, 2, 5, 7]}
    adult_dag2_DP = {14: [9, 0, 3, 4, 2, 12, 5, 6, 7]}
    credit_FTU = {15: [6]}
    credit_DP = {15: [6, 3]}
    for seed in seeds:
        seed_everything(seed, workers=True)
        experiment_decaf("adult", adult_dag, adult_FTU, "DECAF_adult_FTU")
        experiment_decaf("adult", adult_dag, adult_CF, "DECAF_adult_CT")
        experiment_decaf("adult", adult_dag, adult_DP, "DECAF_adult_DP")
        for beta in betas:
            experiment_decaf("credit", credit_dag, {}, "DECAF_credit" + str(beta), beta)
            experiment_decaf("credit", credit_dag, credit_FTU, "DECAF_credit_FTU" + str(beta), beta)
            experiment_decaf("credit", credit_dag, credit_DP, "DECAF_credit_DP" + str(beta), beta)

