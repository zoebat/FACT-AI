import numpy as np

def ftu(clf, X_synth, idx):
    """ Function to measure Fairness Through Unawareness (FTU) for a given dataset and a classifier.

    Args:
        clf: The classifier for which FTU has to be measured
        X_synth: The dataset for which FTU has to be measured
        idx: The index of the protected variable (binary feature)

    Returns:
        float: The return value. The FTU metric for the given dataset and classifier

    """
    # Prepare a dataset with protected variable A set to 0
    X_A0 = np.copy(X_synth)
    X_A0[:, idx] = 0
    y_A0 = clf.predict_proba(X_A0)[:, 1]

    # Prepare a dataset with protected variable A set to 1
    X_A1 = np.copy(X_synth)
    X_A1[:, idx] = 1
    y_A1 = clf.predict_proba(X_A1)[:, 1]

    # Measure FTU by calculating the difference between the predictions of a downstream classifier for setting A to 1 and 0
    P_A0 = np.average(y_A0)
    P_A1 = np.average(y_A1)
    ftu = np.abs(P_A0-P_A1)
    return ftu


def dp(clf, X_synth, idx):
    """ Function to measure Demographic Parity (DP) for a given dataset and a classifier.

    Args:
        clf: The classifier for which FTU has to be measured
        X_synth: The dataset for which FTU has to be measured
        idx: The index of the protected variable (binary feature)

    Returns:
        float: The return value. The DP metric for the given dataset and classifier

    """
    # y_pred = clf.predict(X_synth)
    y_pred = clf.predict_proba(X_synth)[:, 1]

    # Measure DP in terms of the Total Variation
    # i.e., the difference between the predictions of a downstream classifier in terms of positive to negative ratio between the different classes of protected variable A
    X_synth[:, idx] = np.round(X_synth[:, idx])
    y_A0 = []
    y_A1 = []
    for i in range(len(y_pred)):
        if X_synth[i, idx] == 0:
            y_A0.append(y_pred[i])
        if X_synth[i, idx] == 1:
            y_A1.append(y_pred[i])
    y0 = np.average(y_A0)
    y1 = np.average(y_A1)
    return np.abs(y0 - y1)




