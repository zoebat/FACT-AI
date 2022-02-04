import matplotlib.pyplot as plt

def plot_errorbar(betas, metric_name, results):
    fig = plt.figure()
    for (algo, result) in results.items():
        metric, std = list(zip(*result))
        metric = list(metric)
        yerr = list(std)
        plt.errorbar(betas, metric, yerr=yerr,  capsize=4, capthick=1, label=algo)
    plt.xlabel('Bias ' + r'$\beta$')   
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()
