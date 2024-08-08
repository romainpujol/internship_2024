import numpy as np
import matplotlib.pyplot as plt
import time

from csr import *
from dsr import *

def experiment_normalgamma(n, deb, l):
    np.random.seed(42069)
    distribution = generate_data_normalgamma(n)
    mm = [i for i in range(deb, n-deb)]

    # Initialize time and value storage for each method
    time_df =       [0]*len(mm)
    time_km =       [0]*len(mm)
    time_kmlsdf =   [0]*len(mm)
    time_ls =       [0]*len(mm)
    time_lsdf =     [0]*len(mm)

    value_df =      [0]*len(mm)
    value_km =      [0]*len(mm)
    value_kmlsdf =  [0]*len(mm)
    value_ls =      [0]*len(mm)
    value_lsdf =    [0]*len(mm)

    # Loop over each value of m
    for i, m in enumerate(mm):
        print(f"Running iteration {i+1}/{len(mm)}")

        # Get the starters using Dupacova's method
        tic_df = time.time()
        df = dupacova_forward(distribution, m, l)
        tac_df = time.time() - tic_df
        value_df[i] = df[1]

        # Evaluate Local Search with random starters
        tic_ls = time.time()
        random_starters = np.random.choice(np.arange(n), m, replace=False)
        ls = BestFit(distribution, random_starters, l)
        value_ls[i] = ls.local_search()[0]
        tac_ls = time.time() - tic_ls

        # Evaluate Local Search with Dupacova Forward starters
        tic_lsdf = time.time()
        lsdf = BestFit(distribution, df[0], l)
        tac_lsdf = time.time() - tic_lsdf
        value_lsdf[i] = lsdf.local_search()[0]

        # Evaluate K-Means
        tic_km = time.time()
        value_km[i] = k_means(distribution, m, l=l)
        tac_km = time.time() - tic_km

        # Evaluate K-Means with Local search & Dupacova Forward starters
        tic_kmlsdf = time.time()
        value_kmlsdf[i] = k_means(distribution, m, warmcentroids=distribution.atoms[np.array(list(lsdf.ind_red))], l=l)
        tac_kmlsdf = time.time() - tic_kmlsdf

        # Record the time taken for each method
        time_df[i] =        tac_df
        time_km[i] =        tac_km 
        time_kmlsdf[i] =    tac_kmlsdf + tac_lsdf + tac_df
        time_ls[i] =        tac_ls 
        time_lsdf[i] =      tac_df + tac_df


    return mm, value_df, value_km, value_kmlsdf, value_ls, value_lsdf, time_df, time_km, time_kmlsdf, time_ls, time_lsdf

def plot_results(n, mm, value_df, value_km, value_kmlsdf, value_ls, value_lsdf):
    plt.figure(figsize=(10, 6))
    plt.plot(mm, value_df,      label="Dupacova Forward (DF)")
    plt.plot(mm, value_ls,      label="Local Search (LS)")
    plt.plot(mm, value_km,      label="K-Means (KM)")
    plt.plot(mm, value_lsdf,    label="LS with DF")
    plt.plot(mm, value_kmlsdf,  label="KM with LS & DF")
    plt.xlabel('m')
    plt.ylabel('2-Wasserstein distance')
    plt.legend()
    plt.title(f"Efficiency Comparison, n={n}")
    plt.show()

def plot_times(n, mm, time_df, time_km, time_kmlsdf, time_ls, time_lsdf):
    plt.figure(figsize=(10, 6))
    plt.plot(mm, time_df,       label="Dupacova Forward (DF)")
    plt.plot(mm, time_km,       label="K-Means (KM)")
    plt.plot(mm, time_ls,       label="Local Search (LS)")
    plt.plot(mm, time_lsdf,     label="LS with DF")
    plt.plot(mm, time_kmlsdf,   label="KM with LS & DF")
    plt.xlabel('m')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.title(f"Efficiency Comparison, n={n}")
    plt.show()

def main(plot_results_option=False):
    # Experiment parameters
    n = 150
    deb = 20
    l = 2

    mm, value_df, value_km, value_kmlsdf, value_ls, value_lsdf, time_df, time_km, time_kmlsdf, time_ls, time_lsdf = experiment_normalgamma(n, deb, l)

    # Conditionally plot results
    if plot_results_option:
        plot_results(n, mm, value_df, value_km, value_kmlsdf, value_ls, value_lsdf)
        plot_times(n, mm, time_df, time_km, time_kmlsdf, time_ls, time_lsdf)

if __name__ == "__main__":
    main(plot_results_option=True)
