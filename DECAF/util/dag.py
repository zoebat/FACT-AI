import numpy as np 
import pandas as pd 
import argparse
from pycausal.pycausal import pycausal as pcs
from pycausal import prior as p
from pycausal import search as s

from .adult_data import load as adult_load
from .crime_data import load as crime_load
from .credit_data import load as credit_load


def get_data(dataset_name):
    if dataset_name == 'adult':
        _, _, data, _, _ = adult_load()
        column_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "label",
        ]
        # Which prior did we use for the adult dataset?
        pri = [['age', 'sex'],["workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country"],['label']]
    elif dataset_name == 'credit':
        _, _, data, _, _ = credit_load()
        column_names = [
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
        pri = [['age','ethnicity'],['male', 'debt', 'married', 'bankcustomer', 'educationlevel', 'yearsemployed', 'employed', 'creditscore', 'driverslicense', 'citizen', 'zip', 'income'],['approved']]
    elif dataset_name == 'crime':
        _, _, data, _, _ = crime_load()
        column_names = [
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
        # Chose root nodes randomly, maybe experiment with different root nodes
        pri = [['state', 'population', 'perCapInc','MedRent','NumStreet','LemasTotReqPerPop'], ['county','community','communityname','fold','householdsize','racepctblack','racePctWhite','racePctAsian','racePctHisp','agePct12t21','agePct12t29','agePct16t24','agePct65up','numbUrban','pctUrban','medIncome','pctWWage','pctWFarmSelf','pctWInvInc','pctWSocSec','pctWPubAsst','pctWRetire','medFamInc','whitePerCap','blackPerCap','indianPerCap','AsianPerCap','OtherPerCap','HispPerCap','NumUnderPov','PctPopUnderPov','PctLess9thGrade','PctNotHSGrad','PctBSorMore','PctUnemployed','PctEmploy','PctEmplManu','PctEmplProfServ','PctOccupManu','PctOccupMgmtProf','MalePctDivorce','MalePctNevMarr','FemalePctDiv','TotalPctDiv','PersPerFam','PctFam2Par','PctKids2Par','PctYoungKids2Par','PctTeen2Par','PctWorkMomYoungKids','PctWorkMom','NumIlleg','PctIlleg','NumImmig','PctImmigRecent','PctImmigRec5','PctImmigRec8','PctImmigRec10','PctRecentImmig','PctRecImmig5','PctRecImmig8','PctRecImmig10','PctSpeakEnglOnly','PctNotSpeakEnglWell','PctLargHouseFam','PctLargHouseOccup','PersPerOccupHous','PersPerOwnOccHous','PersPerRentOccHous','PctPersOwnOccup','PctPersDenseHous','PctHousLess3BR','MedNumBR','HousVacant','PctHousOccup','PctHousOwnOcc','PctVacantBoarded','PctVacMore6Mos','MedYrHousBuilt','PctHousNoPhone','PctWOFullPlumb','OwnOccLowQuart','OwnOccMedVal','OwnOccHiQuart','RentLowQ','RentMedian','RentHighQ','MedRentPctHousInc','MedOwnCostPctInc','MedOwnCostPctIncNoMtg','NumInShelters','PctForeignBorn','PctBornSameState','PctSameHouse85','PctSameCity85','PctSameState85','LemasSwornFT','LemasSwFTPerPop','LemasSwFTFieldOps','LemasSwFTFieldPerPop','LemasTotalReq','PolicReqPerOffic','PolicPerPop','RacialMatchCommPol','PctPolicWhite','PctPolicBlack','PctPolicHisp','PctPolicAsian','PctPolicMinor','OfficAssgnDrugUnits','NumKindsDrugsSeiz','PolicAveOTWorked','LandArea','PopDens','PctUsePubTrans','PolicCars','PolicOperBudg','LemasPctPolicOnPatr','LemasGangUnitDeploy','LemasPctOfficDrugUn','PolicBudgPerPop'],['ViolentCrimesPerPop']]

        return data, column_names, pri

def find_dag(dataset_name):
    if dataset_name in ['adult', 'credit', 'crime']:
        data, column_names, pri = get_data(dataset_name)
    else:
        raise ValueError("use one of the following three: 'adult', 'credit', 'crime'")

    pc = pcs()
    pc.start_vm()
    prior = p.knowledge(addtemporal = pri)
    tetrad = s.tetradrunner()
    tetrad.run(algoId = 'fges', scoreId = 'cg-bic-score', dfs = data, priorKnowledge = prior,maxDegree = -1, faithfulnessAssumed = True, verbose = False)
    tetrad.getEdges()
    print(tetrad.getTetradGraph())
    dag_seed = []
    for edge in tetrad.getEdges():
        dag_seed.append(list([column_names.index(edge.split(' ')[0]), column_names.index(edge.split(' ')[-1])]))
    return dag_seed
