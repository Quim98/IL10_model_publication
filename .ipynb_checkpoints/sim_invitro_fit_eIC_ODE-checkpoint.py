import numpy as np
from src.models.IL10_seq import model_function
#from src.models.IL10_RAp_seq import model_function
import pandas as pd
from statistics import median
from multiprocessing import Pool
import matplotlib.pyplot as plt
from src.simulate_func import set_simulation,simulate,simulate_parallel
from time import time

start_time = time()
# Import datasets
df_IC_data = pd.read_csv('data/expression/whole_dataset_IL10.tsv.gz', sep="\t", compression="gzip")
df_bind = pd.read_csv('data/binding/IL10_data_param_ABC_SMC_IL10_Control.csv')
#df_bind = pd.read_csv('data/binding/IL10_data_param_ABC_SMC_IL10_RAp.csv')
df_sim_data =  pd.read_csv('data/signaling/IL10_STAT_data.tsv.gz', sep="\t", compression="gzip")

variants = ["WT","Super-10","R5A11D"] #BORRAR!!!!!!!!!!!!!!!!!!
df_sim_data = df_sim_data.loc[df_sim_data["Variant"].isin(variants)]

# Create dataset of simulations
num_sim = 16
df_sim = set_simulation(df_bind, df_sim_data, df_IC_data, num_sim)

# Add pSTAT1 simulation of CD4 and CD8 T cells
df_Tcell = df_sim.loc[df_sim["Cell_type"].isin(["T4.Mean","T8.Mean"])]
df_Tcell["STAT_type"] = "pSTAT1"
df_Tcell["Plot"] = df_Tcell["Plot"].replace(df_Tcell["Plot"].drop_duplicates().values[0],df_sim["Plot"].drop_duplicates().values[-1]+1)
df_Tcell["Plot"] = df_Tcell["Plot"].replace(df_Tcell["Plot"].drop_duplicates().values[1],df_sim["Plot"].drop_duplicates().values[-1]+2)
df_sim = pd.concat([df_sim, df_Tcell], ignore_index=True)
df_sim["Plot OG"] = df_sim["Plot"]

# IC per cell can have variability so we simulate over a normal distribution of IC for the receptors
N = 500  # number of times a single dose-response curve is simulated
df_sim_copy = df_sim.copy()
for i in range(1,N):
    df_sim_e = df_sim_copy.copy()
    df_sim_e["RA0"] = df_sim_e["RA0"]+np.random.normal(loc=0, scale=0.25) # Errror can be adjusted per gene
    df_sim_e["RB0"] = df_sim_e["RB0"]+np.random.normal(loc=0, scale=0.25) # Errror can be adjusted per gene
    df_sim_e["Plot"] = (df_sim_copy["Plot"] + df_sim_copy["Plot"].max()*i).astype(int)
    df_sim = pd.concat([df_sim, df_sim_e], ignore_index=True)

# Simulate
num_cores = 24
df_res = simulate(df_sim, num_cores, model_function)
df_res.to_csv("results/fit_param/simulations_fit_IL10_Control_eIC.csv",index=False)
#df_res.to_csv("results/fit_param/simulations_fit_IL10_RAp_eIC.csv",index=False)


end_time = time()  # Record the end time
execution_time = end_time - start_time  # Calculate execution time
print(f"Execution Time: {execution_time:.6f} seconds")
