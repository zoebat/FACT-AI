import numpy as np
import matplotlib.pyplot as plt

def plot_errorbar(betas, metric_name, results):
    fig = plt.figure()
    for (algo, result) in results.items():
        # metric, low, high = list(zip(*result))
        # metric = list(metric)
        # yerr = [list(low), list(high)]
        plt.errorbar(betas, result, yerr=None,  capsize=4, capthick=1, label=algo)
    plt.xlabel('Bias ' + r'$\beta$')   
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

def get_results(betas, algos):
    results = {}
    n = len(betas)
    for algo in algos:
        precision = np.random.uniform(low=0.25, high=0.9, size=n)
        low = np.random.uniform(low=0.0, high=0.1, size=n)
        high = np.random.uniform(low=0.0, high=0.1, size=n)
        low = precision - low
        high = precision + high 
        results[algo]= list(zip(precision, low, high))
    return results


if __name__ == "__main__":
    n = 6
    betas = np.linspace(0, 1, n, endpoint=True)
    # results = get_results(betas, algos=['AlgoA', 'AlgoB', 'AlgoC'])
    results_ND = np.load("../results_ND2.npy")
    results_FTU = np.load("../results_FTU2.npy")
    results_DP = np.load("../results_DP2.npy")
    precision_ND = []
    precision_FTU = []
    precision_DP = []
    for i in range(len(betas)):
        precision_ND.append(results_ND[i][3])
        precision_FTU.append(results_FTU[i][3])
        precision_DP.append(results_DP[i][3])
    results_precision = {"DECAF-ND": precision_ND,
                        "DECAF-FTU": precision_FTU,
                        "DECAF-DP": precision_DP}
    plot_errorbar(betas, 'FTU', results_precision)
