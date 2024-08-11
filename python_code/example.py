import numpy as np
import matplotlib.pyplot as plt
import time

from csr import *
from dsr import *

# Define each method separately
def run_dupacova(distribution, m, l):
    tic = time.time()
    result = dupacova_forward(distribution, m, l)
    toc = time.time() - tic
    return result, toc

def run_local_search(distribution, m, l, warm_start=None):
    tic = time.time()
    if warm_start is None:
        starters = np.random.choice(np.arange(len(distribution)), m, replace=False)
    else:
        starters = warm_start
    ls = BestFit(distribution, starters, l)
    ls.local_search()
    toc = time.time() - tic
    return ls.get_distance(), toc

def run_k_means(distribution, m, l, warm_start=None):
    tic = time.time()
    if warm_start is None:
        result = k_means(distribution, m, l=l)
    else:
        result = k_means(distribution, m, warmcentroids=distribution.atoms[warm_start], l=l)
    toc = time.time() - tic
    return result, toc

def run_milp(distribution, m):
    tic = time.time()
    result = milp(distribution, m)
    toc = time.time() - tic
    return result, toc

# Experiment function
def experiment_normalgamma(n, deb, l, methods_to_run):
    distribution = generate_data_normalgamma(n)
    mm = [i for i in range(10, 100, 10)]

    # Initialize storage
    results = {method: {'values': [], 'times': []} for method in methods_to_run}

    # Loop over each value of m
    for i, m in enumerate(mm):
        print(f"Running iteration {i+1}/{len(mm)}")

        warm_start = None
        if 'Dupacova' in methods_to_run:
            df_result, df_time = run_dupacova(distribution, m, l)
            results['Dupacova']['values'].append(df_result[1])
            results['Dupacova']['times'].append(df_time)
            warm_start = df_result[0]  # Use the Dupacova starters for warm-starts

        if 'Local Search' in methods_to_run:
            ls_value, ls_time = run_local_search(distribution, m, l)
            results['Local Search']['values'].append(ls_value)
            results['Local Search']['times'].append(ls_time)

        if 'LS with Dupacova' in methods_to_run and warm_start is not None:
            lsdf_value, lsdf_time = run_local_search(distribution, m, l, warm_start=warm_start)
            results['LS with Dupacova']['values'].append(lsdf_value)
            results['LS with Dupacova']['times'].append(lsdf_time + df_time)  # Include Dupacova time

        if 'K-Means' in methods_to_run:
            km_value, km_time = run_k_means(distribution, m, l)
            results['K-Means']['values'].append(km_value)
            results['K-Means']['times'].append(km_time)

        if 'KM with Dupacova' in methods_to_run and warm_start is not None:
            kmlsdf_value, kmlsdf_time = run_k_means(distribution, m, l, warm_start=warm_start)
            results['KM with Dupacova']['values'].append(kmlsdf_value)
            results['KM with Dupacova']['times'].append(kmlsdf_time + df_time)  # Include Dupacova time

        if 'MILP' in methods_to_run:
            milp_value, milp_time = run_milp(distribution, m)
            results['MILP']['values'].append(milp_value)
            results['MILP']['times'].append(milp_time)

    return mm, results

def plot_results(n, mm, results):
    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        plt.plot(mm, data['values'], label=method)

    plt.xlabel('Number of reduced atoms')
    plt.ylabel('2-Wasserstein distance')
    plt.legend()
    plt.title(f"Efficiency Comparison, n={n}")
    plt.show()

def plot_times(n, mm, results):
    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        plt.plot(mm, data['times'], label=method)

    plt.xlabel('Number of reduced atoms')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.title(f"Efficiency Comparison, n={n}")
    plt.show()

def main(plot_results_option=False):
    # Experiment parameters
    n = 200
    deb = 10
    l = 2

    # Define methods to run
    methods_to_run = [
        'Dupacova',
        'Local Search',
        'LS with Dupacova',
        'K-Means',
        'KM with Dupacova',
        'MILP'
    ]

    mm, results = experiment_normalgamma(n, deb, l, methods_to_run)

    # Conditionally plot results
    if plot_results_option:
        plot_results(n, mm, results)
        plot_times(n, mm, results)

if __name__ == "__main__":
    main(plot_results_option=True)
