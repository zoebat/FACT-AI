import numpy as np 
import pandas as pd 
import argparse
from pycausal.pycausal import pycausal as pcs
from pycausal import prior as p
from pycausal import search as s
from credit_data import credit_load

def find_dag(data, column_names, pri):
    pc = pcs()
    pc.start_vm()
    prior = p.knowledge(addtemporal = pri)
    tetrad = s.tetradrunner()
    tetrad.run(algoId = 'fges', scoreId = 'cg-bic-score', dfs = data, priorKnowledge = prior,
           maxDegree = -1, faithfulnessAssumed = True, verbose = False)
    tetrad.getEdges()
    dag_seed = []
    for edge in tetrad.getEdges():
        dag_seed.append(list([column_names.index(edge.split(' ')[0]), column_names.index(edge.split(' ')[-1])]))
    
    return dag_seed


if __name__ == "__main__":
    _, _, _, df = credit_load()
    parser = argparse.ArgumentParser()  # Parse training configuration

    parser.add_argument('--data', type=pd.DataFrame, default = df, help="Data to find dag for")
    parser.add_argument('--column_names', type=list, default=[
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
        ], help='List with all the column names')
    parser.add_argument('--pri', type=list, default=[['age','ethnicity'],['male', 'debt', 'married', 'bankcustomer', 'educationlevel', 'yearsemployed',
                'employed', 'creditscore', 'driverslicense', 'citizen', 'zip', 'income'],['approved']], 
                help='Prior Knowledge')

    args = parser.parse_args()
    kwargs = vars(args)

    dag_seed = find_dag(**kwargs)
    print(dag_seed)
