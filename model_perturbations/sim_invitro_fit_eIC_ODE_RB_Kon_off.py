# In this script we see the effects of chaging the Kon vs Koff in an IL-10RB mutant

import sys
sys.path.append('/users/lserrano/qmarti/PhD_code/IL10') # Add the directory containing the script to sys.path
import numpy as np
from src.models.IL10_RAp_ODE import model_function
import pandas as pd
from statistics import median
from multiprocessing import Pool
import matplotlib.pyplot as plt
from src.simulate_func import set_simulation_cell_list,simulate_parallel_ODE
from time import time

start_time = time()
# Import datasets
path_data = '/users/lserrano/qmarti/PhD_code'
df_IC_data = pd.read_csv(path_data+'/IL10/data/expression/whole_dataset_IL10.tsv.gz', sep="\t", compression="gzip")
df_bind = pd.read_csv(path_data+'/IL10/data/binding/IL10_data_param_ABC_SMC_IL10_RAp_ODE.csv')

# Get mutant with x10 lower unbinding rate
df_bind = df_bind.loc[df_bind["Variant"].isin(["WT"])]
df_bind.loc[len(df_bind)] = df_bind.loc[df_bind["Variant"]=="WT"].values[0]
df_bind.loc[len(df_bind)-1,"Variant"] = "MutRB_koff"
df_bind.loc[df_bind["Variant"]=="MutRB_koff","k_IL_RB_b"] = df_bind.loc[df_bind["Variant"]=="MutRB_koff","k_IL_RB_b"]/10
df_bind.loc[df_bind["Variant"]=="MutRB_koff","k_IL_RA_RB_b"] = df_bind.loc[df_bind["Variant"]=="MutRB_koff","k_IL_RA_RB_b"]/10
df_bind.loc[len(df_bind)] = df_bind.loc[df_bind["Variant"]=="WT"].values[0]
df_bind.loc[len(df_bind)-1,"Variant"] = "MutRB_kon"
df_bind.loc[df_bind["Variant"]=="MutRB_kon","k_IL_RB_f"] = df_bind.loc[df_bind["Variant"]=="MutRB_kon","k_IL_RB_f"]*10
df_bind.loc[df_bind["Variant"]=="MutRB_kon","k_IL_RA_RB_f"] = df_bind.loc[df_bind["Variant"]=="MutRB_kon","k_IL_RA_RB_f"]*10

# Create dataset of simulations
num_sim = 12
cell_list = ["ACH-000786","T8.Mean"]
variant_list = ["WT","MutRB_koff","MutRB_kon"]
STAT_list = ["pSTAT3","pSTAT1"]
low_IL = 1e-13
high_IL = 1e-7
df_sim = set_simulation_cell_list(df_bind, df_IC_data, cell_list, variant_list, STAT_list, num_sim, low_IL, high_IL)
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

# Simulate mutation affects koff
num_cores = 24
df_res = simulate_parallel_ODE(df_sim, num_cores, model_function)
df_res.to_csv("../results/model_perturbations/reviews/simulations_fit_IL10_RAp_eIC_ODE_RB_Kon_off.csv",index=False)

end_time = time()  # Record the end time
execution_time = end_time - start_time  # Calculate execution time
print(f"Execution Time: {execution_time:.6f} seconds")
