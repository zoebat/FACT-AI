import numpy as np
from numpy.lib.function_base import median
import pandas as pd
from sklearn import preprocessing

def load():
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
    names = [
        'state',
        'county',
        'community',
        'communityname',
        'fold',
        'population',
        'householdsize',
        'racepctblack',
        'racePctWhite',
        'racePctAsian',
        'racePctHisp',
        'agePct12t21',
        'agePct12t29',
        'agePct16t24',
        'agePct65up',
        'numbUrban',
        'pctUrban',
        'medIncome',
        'pctWWage',
        'pctWFarmSelf',
        'pctWInvInc',
        'pctWSocSec',
        'pctWPubAsst',
        'pctWRetire',
        'medFamInc',
        'perCapInc',
        'whitePerCap',
        'blackPerCap',
        'indianPerCap',
        'AsianPerCap',
        'OtherPerCap',
        'HispPerCap',
        'NumUnderPov',
        'PctPopUnderPov',
        'PctLess9thGrade',
        'PctNotHSGrad',
        'PctBSorMore',
        'PctUnemployed',
        'PctEmploy',
        'PctEmplManu',
        'PctEmplProfServ',
        'PctOccupManu',
        'PctOccupMgmtProf',
        'MalePctDivorce',
        'MalePctNevMarr',
        'FemalePctDiv',
        'TotalPctDiv',
        'PersPerFam',
        'PctFam2Par',
        'PctKids2Par',
        'PctYoungKids2Par',
        'PctTeen2Par',
        'PctWorkMomYoungKids',
        'PctWorkMom',
        'NumIlleg',
        'PctIlleg',
        'NumImmig',
        'PctImmigRecent',
        'PctImmigRec5',
        'PctImmigRec8',
        'PctImmigRec10',
        'PctRecentImmig',
        'PctRecImmig5',
        'PctRecImmig8',
        'PctRecImmig10',
        'PctSpeakEnglOnly',
        'PctNotSpeakEnglWell',
        'PctLargHouseFam',
        'PctLargHouseOccup',
        'PersPerOccupHous',
        'PersPerOwnOccHous',
        'PersPerRentOccHous',
        'PctPersOwnOccup',
        'PctPersDenseHous',
        'PctHousLess3BR',
        'MedNumBR',
        'HousVacant',
        'PctHousOccup',
        'PctHousOwnOcc',
        'PctVacantBoarded',
        'PctVacMore6Mos',
        'MedYrHousBuilt',
        'PctHousNoPhone',
        'PctWOFullPlumb',
        'OwnOccLowQuart',
        'OwnOccMedVal',
        'OwnOccHiQuart',
        'RentLowQ',
        'RentMedian',
        'RentHighQ',
        'MedRent',
        'MedRentPctHousInc',
        'MedOwnCostPctInc',
        'MedOwnCostPctIncNoMtg',
        'NumInShelters',
        'NumStreet',
        'PctForeignBorn',
        'PctBornSameState',
        'PctSameHouse85',
        'PctSameCity85',
        'PctSameState85',
        'LemasSwornFT',
        'LemasSwFTPerPop',
        'LemasSwFTFieldOps',
        'LemasSwFTFieldPerPop',
        'LemasTotalReq',
        'LemasTotReqPerPop',
        'PolicReqPerOffic',
        'PolicPerPop',
        'RacialMatchCommPol',
        'PctPolicWhite',
        'PctPolicBlack',
        'PctPolicHisp',
        'PctPolicAsian',
        'PctPolicMinor',
        'OfficAssgnDrugUnits',
        'NumKindsDrugsSeiz',
        'PolicAveOTWorked',
        'LandArea',
        'PopDens',
        'PctUsePubTrans',
        'PolicCars',
        'PolicOperBudg',
        'LemasPctPolicOnPatr',
        'LemasGangUnitDeploy',
        'LemasPctOfficDrugUn',
        'PolicBudgPerPop',
        'ViolentCrimesPerPop'
    ]

    df = pd.read_csv(path, names=names, index_col=False)
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    df.reset_index(drop=True, inplace=True)

    # drop all columns that have an entry with ?
    for col in df:
        if df[col].dtype == 'object':
            df.drop(col, axis=1, inplace=True)

    df = df.dropna(how = 'all')

    median_target = df['ViolentCrimesPerPop'].median()

    # binarize target variable
    df['ViolentCrimesPerPop'] = np.where(df['ViolentCrimesPerPop'] > median_target, 1, 0)

    # binarize racepctblack
    df['racepctblack'] = np.where(df['racepctblack'] > 0.5, 1, 0)

    # binarize racePctWhite
    df['racePctWhite'] = np.where(df['racePctWhite'] > 0.5, 1, 0)

    # binarize racePctAsian
    df['racePctAsian'] = np.where(df['racePctAsian'] > 0.5, 1, 0)

    # binarize racePctHisp
    df['racePctHisp'] = np.where(df['racePctHisp'] > 0.5, 1, 0)

    dfr = df.copy()
    df = df.values

    min_max_scaler = preprocessing.MinMaxScaler()
    Xy = min_max_scaler.fit_transform(df)

    return dfr, Xy, min_max_scaler