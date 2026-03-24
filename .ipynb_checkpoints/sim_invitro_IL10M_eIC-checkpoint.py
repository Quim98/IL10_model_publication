import numpy as np
#from src.models.IL10_ODE import model_function
from src.models.IL10_RAp_ODE import model_function
import pandas as pd
from statistics import median
from multiprocessing import Pool
import matplotlib.pyplot as plt
from src.simulate_func import set_simulation,simulate_parallel_ODE
from time import time

start_time = time()
# Import datasets
df_IC_data = pd.read_csv('data/expression/whole_dataset_IL10.tsv.gz', sep="\t", compression="gzip")
#df_bind = pd.read_csv('data/binding/IL10_data_param_ABC_SMC_IL10_Control.csv')
df_bind = pd.read_csv('data/binding/IL10_data_param_ABC_SMC_IL10_RAp.csv')
#df_bind = pd.read_csv('data/binding/IL10_data_param_ABC_SMC_IL10_RAp_ODE_full.csv')
df_sim_data = pd.concat([pd.read_csv('data/signaling/IL10_STAT_data.tsv.gz', sep="\t", compression="gzip"), pd.read_csv('data/signaling/IL10_STAT_data_CD8_Gorby.tsv.gz', sep="\t", compression="gzip")], ignore_index=True)

###############################################################################################################
variants = ["WT","Super-10","R5A11D"]
df_sim_data = df_sim_data.loc[df_sim_data["Variant"].isin(variants)]
###############################################################################################################

# Create dataset of simulations
num_sim = 20
df_sim = set_simulation(df_bind, df_sim_data, df_IC_data, num_sim)

df_Tcell = df_sim.loc[df_sim["Plot"].isin([8,9])]
df_Tcell["STAT_type"] = "pSTAT1"
df_Tcell["Plot"] = df_Tcell["Plot"].replace(df_Tcell["Plot"].drop_duplicates().values[0],12)
df_Tcell["Plot"] = df_Tcell["Plot"].replace(df_Tcell["Plot"].drop_duplicates().values[1],13)
df_sim = pd.concat([df_sim, df_Tcell], ignore_index=True)

# Simulate
num_cores = 24
df_res = simulate_parallel_ODE(df_sim, num_cores, model_function)
#df_res.to_csv("results/fit_param/simulations_fit_IL10_Control_ODE.csv",index=False)
df_res.to_csv("results/fit_param/simulations_fit_IL10_RAp_ODE_2.csv",index=False)


end_time = time()  # Record the end time
execution_time = end_time - start_time  # Calculate execution time
print(f"Execution Time: {execution_time:.6f} seconds")
