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
    y_A0 = clf.predict(X_A0)

    # Prepare a dataset with protected variable A set to 1
    X_A1 = np.copy(X_synth)
    X_A1[:, idx] = 1
    y_A1 = clf.predict(X_A1)

    # Measure FTU by calculating the difference between the predictions of a downstream classifier for setting A to 1 and 0
    ftu = np.abs(y_A0 - y_A1)
    return np.average(ftu)


def dp(clf, X_synth, idx):
    """ Function to measure Demographic Parity (DP) for a given dataset and a classifier.

    Args:
        clf: The classifier for which FTU has to be measured
        X_synth: The dataset for which FTU has to be measured
        idx: The index of the protected variable (binary feature)

    Returns:
        float: The return value. The DP metric for the given dataset and classifier

    """
    y_pred = clf.predict(X_synth)

    # Measure DP in terms of the Total Variation
    # i.e., the difference between the predictions of a downstream classifier in terms of positive to negative ratio between the different classes of protected variable A
    X_synth[:, idx] = np.round(X_synth[:, idx])
    y_A0 = np.where(X_synth[:, idx] == 0, y_pred, 0)
    y_A1 = np.where(X_synth[:, idx] == 1, y_pred, 0)
    y0 = np.average(y_A0)
    y1 = np.average(y_A1)
    return np.abs(y0 - y1)




