################################################################################
######################## A simple 2D scalable example ##########################
################################################################################
#
# The following example aims to reduce a n atoms distribution into a distribution 
# with different values of m < n. The atoms of the input distribution are only
# 2-Dimensional (one dimension is drawn from a Normal distribution and the other
# from a Gamma distribution), but both the number of atoms n of the original
# distribution and the number of atoms m of the reduced distribution can be
# changed. 
#
# Every Scenario Reduction method in both of dsr.py and csr.py are tested. See
# these two files for more details on each methods mentioned.
#
# Even though only 2-Dimensional, we can use this 2D scalable (in n and m)
# example to see experimental trends (that support theorical results)
# on the behaviour of the tested methods. Both in the "quality" of the output
# through the l-Wasserstein distance W_l(P, Q) between the original distribution
# P and the reduced distribution Q, but also in the "CPU time" needed to compute Q.
# 
# In the main function just below one may comment the methods that should not be
# run. The plot functions then adapt accordingly to only plot the methods that
# were ran.
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import time

from csr import *
from dsr import *

################################################################################
############# Customizable run of methods of dsr.py or csr.py ##################
################################################################################
#
# Simply edit the list "indices_to_run" to change the methods that need to be
# run. There is a dependency of methods that have "... with Dupacova" in their
# name: they need Dupacova's algorithm (index 0) to be run.
#
# Data n and "m" can also be changed, n directly and m through the "deb" parameter.
#
################################################################################

def main(plot_results_option=False):
    # Experiment parameters
    n = 200
    deb = 10
    l = 2

    list_of_methods = [
        'Dupacova',             # 0
        'Best Fit',             # 1
        'BF with Dupacova',     # 2
        'First Fit',            # 3
        'FF with Dupacova',     # 4
        'FF with shuffle',      # 5
        'FFS with Dupacova',    # 6
        'K-Means',              # 7
        'KM with Dupacova',     # 8
        'MILP'                  # 9
    ]

    # SELECT THE INDEXES OF THE METHODS YOU WANT TO BE RUN
    # METHODS "...WITH DUPACOVA" NEED DUPACOVA TO BE RUN (index 0)
    indices_to_run = [0, 2, 4, 5, 6, 9]
    ###
    
    mm, results = experiment_normalgamma(n, deb, l, [list_of_methods[i] for i in indices_to_run])

    # Conditionally plot results
    if plot_results_option:
        plot_results(n, mm, results)
        plot_times(n, mm, results)

################################################################################

# Define each method separately
def run_dupacova(distribution, m, l):
    tic = time.time()
    result = dupacova_forward(distribution, m, l)
    toc = time.time() - tic
    return result, toc

def run_bestfit(distribution, m, l, warm_start=None):
    tic = time.time()
    if warm_start is None:
        starters = np.random.choice(np.arange(len(distribution)), m, replace=False)
    else:
        starters = warm_start
    bf = BestFit(distribution, starters, l)
    bf.local_search()
    toc = time.time() - tic
    return bf.get_distance(), toc

def run_firstfit(distribution, m, l, warm_start=None, shuffle=False):
    tic = time.time()
    if warm_start is None:
        starters = np.random.choice(np.arange(len(distribution)), m, replace=False)
    else:
        starters = warm_start
    ff = FirstFit(distribution, starters, l, shuffle=shuffle)
    ff.local_search()
    toc = time.time() - tic
    return ff.get_distance(), toc

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

        if 'Best Fit' in methods_to_run:
            bf_value, bf_time = run_bestfit(distribution, m, l)
            results['Best Fit']['values'].append(bf_value)
            results['Best Fit']['times'].append(bf_time)

        if 'BF with Dupacova' in methods_to_run and warm_start is not None:
            bfdf_value, bfdf_time = run_bestfit(distribution, m, l, warm_start=warm_start)
            results['BF with Dupacova']['values'].append(bfdf_value)
            results['BF with Dupacova']['times'].append(bfdf_time + df_time)  # Include Dupacova time

        if 'First Fit' in methods_to_run:
            ff_value, ff_time = run_firstfit(distribution, m, l)
            results['First Fit']['values'].append(ff_value)
            results['First Fit']['times'].append(ff_time)

        if 'FF with Dupacova' in methods_to_run:
            ff_value, ff_time = run_firstfit(distribution, m, l, warm_start=warm_start)
            results['FF with Dupacova']['values'].append(ff_value)
            results['FF with Dupacova']['times'].append(ff_time)

        if 'FF with shuffle' in methods_to_run:
            ffs_value, ffs_time = run_firstfit(distribution, m, l, shuffle=True)
            results['FF with shuffle']['values'].append(ffs_value)
            results['FF with shuffle']['times'].append(ffs_time)

        if 'FFS with Dupacova' in methods_to_run:
            ffsd_value, ffsd_time = run_firstfit(distribution, m, l, shuffle=True, warm_start=warm_start)
            results['FFS with Dupacova']['values'].append(ffsd_value)
            results['FFS with Dupacova']['times'].append(ffsd_time)

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

def plot_results(n, mm, results, filename="2dim_value.pdf"):
    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        plt.plot(mm, data['values'], label=method)

    plt.xlabel("Number of reduced atoms")
    plt.ylabel("2-Wasserstein distance")
    plt.legend()
    plt.title(f"Efficiency Comparison, n={n}")

    # Save the plot to a PDF file
    plt.savefig(filename)
    plt.close()  

def plot_times(n, mm, results, filename="2dim_time.pdf"):
    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        plt.plot(mm, data['times'], label=method)

    plt.xlabel("Number of reduced atoms")
    plt.ylabel("Time (s)")
    plt.legend()
    
    # Save the plot to a PDF file
    plt.savefig(filename)
    plt.close()  

if __name__ == "__main__":
    main(plot_results_option=True)
