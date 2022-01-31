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
        'communityname', # string
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

    for col in df:
        if df[col].dtype == 'object':
            df = df[df[col] != "?"]
            # number of entries: 1994 -> 123

    df = df.dropna(how = 'all')

    df['communityname'] = preprocessing.LabelEncoder().fit_transform(df['communityname'])

    median_target = df['ViolentCrimesPerPop'].median()

    # binarize target variable
    df['ViolentCrimesPerPop'] = np.where(df['ViolentCrimesPerPop'] > median_target, 1, 0)
    # print(df)

    dfr = df.copy()
    df = df.values
    X = df[:, :127].astype(np.float32)
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    y = df[:, 127].astype(np.uint32)
    Xy = min_max_scaler.fit_transform(df)

    return X, y, dfr, Xy, min_max_scaler



    # print(df)
    # print(df.info()) # dus 123 entries, en 128 attributes

    # <class 'pandas.core.frame.DataFrame'>
    # Int64Index: 123 entries, 16 to 1992
    # Columns: 128 entries, state to ViolentCrimesPerPop
    # dtypes: float64(100), int64(3), object(25)
    # memory usage: 124.0+ KB