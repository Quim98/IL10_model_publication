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
df_bind = pd.read_csv('data/binding/IL10_data_param_ABC_SMC_IL10_RAp_ODE.csv')
df_bind.loc[df_bind["Variant"]=="WT","k_IL_RA_RB_b"] = df_bind.loc[df_bind["Variant"]=="WT","k_IL_RA_RB_b"]*25
df_sim_data = pd.read_csv('data/signaling/IL10M_STAT_data_Gorby.tsv.gz', sep="\t", compression="gzip")

# Create dataset of simulations
num_sim = 20
df_sim = set_simulation(df_bind, df_sim_data, df_IC_data, num_sim)

# Simulate
num_cores = 24
df_res = simulate_parallel_ODE(df_sim, num_cores, model_function)
df_res.to_csv("results/fit_param/simulations_fit_IL10_RAp_ODE_MONO.csv",index=False)


end_time = time()  # Record the end time
execution_time = end_time - start_time  # Calculate execution time
print(f"Execution Time: {execution_time:.6f} seconds")
