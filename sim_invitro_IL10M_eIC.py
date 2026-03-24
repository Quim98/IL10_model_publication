# Script predicting pSTAT1 and pSTAT3 among monomeric variants of IL-10, given certain error in the initial conditions. These results are used in the prediction of dose-response curves.

import numpy as np
from src.models.IL10M_RAp_ODE import model_function
import pandas as pd
from statistics import median
from multiprocessing import Pool
import matplotlib.pyplot as plt
from src.simulate_func import set_simulation,simulate_parallel_ODE
from time import time

start_time = time()
# Import datasets
df_IC_data = pd.read_csv('data/expression/whole_dataset_IL10.tsv.gz', sep="\t", compression="gzip")
#df_bind = pd.read_csv('data/binding/IL10_data_param_ABC_SMC_IL10_RAp_ODE.csv')
df_bind = pd.read_csv('data/binding/IL10_data_param_ABC_SMC_IL10_RAp_ODE_IL10M.csv')
df_sim_data = pd.read_csv('data/signaling/IL10M_STAT_data_Gorby.tsv.gz', sep="\t", compression="gzip")

# Create dataset of simulations
num_sim = 20
df_sim = set_simulation(df_bind, df_sim_data, df_IC_data, num_sim)
df_sim["Plot OG"] = df_sim["Plot"]

# IC per cell can have variability so we simulate over a normal distribution of IC for the receptors
N = 200  # number of times a single dose-response curve is simulated
df_sim_copy = df_sim.copy()
for i in range(1,N):
    df_sim_e = df_sim_copy.copy()
    df_sim_e["RA0"] = df_sim_e["RA0"]+np.random.normal(loc=0, scale=0.2) # Errror can be adjusted per gene
    df_sim_e["RB0"] = df_sim_e["RB0"]+np.random.normal(loc=0, scale=0.2) # Errror can be adjusted per gene
    df_sim_e["Plot"] = (df_sim_copy["Plot"] + df_sim_copy["Plot"].max()*i).astype(int)
    df_sim = pd.concat([df_sim, df_sim_e], ignore_index=True)

# Simulate
num_cores = 22
df_res = simulate_parallel_ODE(df_sim, num_cores, model_function)
#df_res.to_csv("results/fit_param/simulations_fit_IL10_RAp_ODE_MONO_eIC.csv",index=False)
#df_res.to_csv("results/fit_param/simulations_fit_IL10_RAp_ODE_MONO_fit_eIC.csv",index=False)

end_time = time()  # Record the end time
execution_time = end_time - start_time  # Calculate execution time
print(f"Execution Time: {execution_time:.6f} seconds")
