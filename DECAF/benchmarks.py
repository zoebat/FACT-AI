import os
import numpy as np
import pytorch_lightning as pl
import torch

from sklearn.metrics import precision_score, recall_score, roc_auc_score
from util import adult_data, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from util import data
from models.gan import GAN
from models.wgan import WGAN



def experiment_benchmark_gan(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    baseline_clf = MLPClassifier().fit(X_train, y_train)
    y_pred = baseline_clf.predict(X_test)

    print(
        "baseline scores",
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        roc_auc_score(y_test, y_pred),
    )
    
    dm = data.DataModule(np.append(X_train, np.expand_dims(y_train, axis=1), axis=1))


    precisions = []
    recalls = []
    aucs = []
    dps = []
    ftus = []
    
    os.makedirs('checkpoints/GAN', exist_ok=True)
    for i in range(10):
        model = GAN(data_dim=dm.dims[0])
        trainer = pl.Trainer(max_epochs=50, logger=False)
        trainer.fit(model, dm)

        Xy_synth = model.sample(X.shape[0])

        X_synth = Xy_synth[:, :14]
        y_synth = np.round(Xy_synth[:, 14])
    
        synth_clf = MLPClassifier().fit(X_synth, y_synth)
        y_pred_synth = synth_clf.predict(X_test)
        y_pred_synth_proba = synth_clf.predict_proba(X_test)

        dps.append(metrics.dp(synth_clf, X_test, 9))
        ftus.append(metrics.ftu(synth_clf, X_test, 9))
        precisions.append(precision_score(y_test, y_pred_synth))
        recalls.append(recall_score(y_test, y_pred_synth))
        aucs.append(roc_auc_score(y_test, y_pred_synth_proba[:, 1]))
        trainer.save_checkpoint("checkpoints/GAN/GAN{}.ckpt".format(i))

    print("Precision: {}, std: {}".format(np.mean(precisions), np.std(precisions)))
    print("Recall: {}, std: {}".format(np.mean(recalls), np.std(recalls)))
    print("AUC: {}, std: {}".format(np.mean(aucs), np.std(aucs)))
    print("DP: {}, std: {}".format(np.mean(dps), np.std(dps)))
    print("FTU: {}, std: {}".format(np.mean(ftus), np.std(ftus)))

def experiment_benchmark_wgan_gp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    baseline_clf = MLPClassifier().fit(X_train, y_train)
    y_pred = baseline_clf.predict(X_test)

    print(
        "baseline scores",
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        roc_auc_score(y_test, y_pred),
    )
    
    dm = data.DataModule(np.append(X_train, np.expand_dims(y_train, axis=1), axis=1))

    precisions = []
    recalls = []
    aucs = []
    dps = []
    ftus = []
    
    os.makedirs('checkpoints/WGAN', exist_ok=True)
    for i in range(10):
        model = WGAN(data_dim=dm.dims[0], lambda_gp=1)

        trainer = pl.Trainer(max_epochs=50, logger=False)
        trainer.fit(model, dm)
        Xy_synth = model.sample(X.shape[0])

        X_synth = Xy_synth[:, :14]
        y_synth = np.round(Xy_synth[:, 14])
    
        synth_clf = MLPClassifier().fit(X_synth, y_synth)
        y_pred_synth = synth_clf.predict(X_test)
        y_pred_synth_proba = synth_clf.predict_proba(X_test)

        dps.append(metrics.dp(synth_clf, X_test, 9))
        ftus.append(metrics.ftu(synth_clf, X_test, 9))
        precisions.append(precision_score(y_test, y_pred_synth))
        recalls.append(recall_score(y_test, y_pred_synth))
        aucs.append(roc_auc_score(y_test, y_pred_synth_proba[:, 1]))
        trainer.save_checkpoint("checkpoints/WGAN/WGAN{}.ckpt".format(i))
    
    print("Precision: {}, std: {}".format(np.mean(precisions), np.std(precisions)))
    print("Recall: {}, std: {}".format(np.mean(recalls), np.std(recalls)))
    print("AUC: {}, std: {}".format(np.mean(aucs), np.std(aucs)))
    print("DP: {}, std: {}".format(np.mean(dps), np.std(dps)))
    print("FTU: {}, std: {}".format(np.mean(ftus), np.std(ftus)))


def experiment_benchmark_gan_pr(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    baseline_clf = MLPClassifier().fit(X_train, y_train)
    y_pred = baseline_clf.predict(X_test)

    print(
        "baseline scores",
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        roc_auc_score(y_test, y_pred),
    )
    
    dm = data.DataModule(np.append(X_train, np.expand_dims(y_train, axis=1), axis=1))


    precisions = []
    recalls = []
    aucs = []
    dps = []
    
    os.makedirs('checkpoints/GAN-PR', exist_ok=True)
    for i in range(10):
        model = GAN(data_dim=dm.dims[0])
        trainer = pl.Trainer(max_epochs=50, logger=False)
        trainer.fit(model, dm)

        Xy_synth = model.sample(X.shape[0])

        X_synth = Xy_synth[:, :14]
        X_synth = np.delete(X_synth, 9, axis=1)
        y_synth = np.round(Xy_synth[:, 14])

        X_test_pr = np.delete(X_test, 9, axis=1)
    
        synth_clf = MLPClassifier().fit(X_synth, y_synth)
        y_pred_synth = synth_clf.predict(X_test_pr)
        y_pred_synth_proba = synth_clf.predict_proba(X_test_pr)

        dps.append(metrics.dp_pr(synth_clf, X_test, 9))
        precisions.append(precision_score(y_test, y_pred_synth))
        recalls.append(recall_score(y_test, y_pred_synth))
        aucs.append(roc_auc_score(y_test, y_pred_synth_proba[:, 1]))
        trainer.save_checkpoint("checkpoints/GAN-PR/GAN-PR{}.ckpt".format(i))

    print("Precision: {}, std: {}".format(np.mean(precisions), np.std(precisions)))
    print("Recall: {}, std: {}".format(np.mean(recalls), np.std(recalls)))
    print("AUC: {}, std: {}".format(np.mean(aucs), np.std(aucs)))
    print("DP: {}, std: {}".format(np.mean(dps), np.std(dps)))


def experiment_benchmark_wgan_pr(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    baseline_clf = MLPClassifier().fit(X_train, y_train)
    y_pred = baseline_clf.predict(X_test)

    print(
        "baseline scores",
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        roc_auc_score(y_test, y_pred),
    )
    
    dm = data.DataModule(np.append(X_train, np.expand_dims(y_train, axis=1), axis=1))


    precisions = []
    recalls = []
    aucs = []
    dps = []
    
    os.makedirs('checkpoints/WGAN-PR', exist_ok=True)
    for i in range(10):
        model = WGAN(data_dim=dm.dims[0])
        trainer = pl.Trainer(max_epochs=50, logger=False)
        trainer.fit(model, dm)

        Xy_synth = model.sample(X.shape[0])

        X_synth = Xy_synth[:, :14]
        X_synth = np.delete(X_synth, 9, axis=1)
        y_synth = np.round(Xy_synth[:, 14])

        X_test_pr = np.delete(X_test, 9, axis=1)
    
        synth_clf = MLPClassifier().fit(X_synth, y_synth)
        y_pred_synth = synth_clf.predict(X_test_pr)
        y_pred_synth_proba = synth_clf.predict_proba(X_test_pr)

        dps.append(metrics.dp_pr(synth_clf, X_test, 9))
        precisions.append(precision_score(y_test, y_pred_synth))
        recalls.append(recall_score(y_test, y_pred_synth))
        aucs.append(roc_auc_score(y_test, y_pred_synth_proba[:, 1]))
        trainer.save_checkpoint("checkpoints/WGAN-PR/WGAN-PR{}.ckpt".format(i))

    print("Precision: {}, std: {}".format(np.mean(precisions), np.std(precisions)))
    print("Recall: {}, std: {}".format(np.mean(recalls), np.std(recalls)))
    print("AUC: {}, std: {}".format(np.mean(aucs), np.std(aucs)))
    print("DP: {}, std: {}".format(np.mean(dps), np.std(dps)))

if __name__ == "__main__":
    X, y, dfr, Xy, min_max_scaler = adult_data.load()
    experiment_benchmark_wgan_pr(X, y)