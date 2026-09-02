# Script infering the IL-10RB binding dynamics among dimeric IL-10 variants (Super-10 and R5A11D) for the Receptor Memory model

# Imports
import sys
sys.path.append('/users/lserrano/qmarti/PhD_code/IL10') # Add the directory containing the script to sys.path
import numpy as np
from src.models.IL10_RAp_ODE import model_function as model_function_IL10_RAp
import pandas as pd
from scipy.optimize import curve_fit
from src.simulate_func import set_simulation_ABC_MS,update_df_sim_bind_ABC_MS,simulate_sequential_ODE,hill_func,model_selection_ABC_SMC
import pyabc
import pyabc.sge
import time

# Initialize datasets
path_data = '/users/lserrano/qmarti/PhD_code'
df_bind = pd.read_csv(path_data+'/IL10/data/binding/IL10_data_param_ABC_SMC_IL10_RAp_ODE.csv')
df_fit_data = pd.read_csv(path_data+'/IL10/data/ABC_SMC/param_inference_IC_IL10_ODE_mut.csv') # Parameters to fit
df_fit_data["Var__Param"] = df_fit_data["Variant"]+"__"+df_fit_data["Parameter"]
df_fit_data = df_fit_data[["Model","Var__Param","Initial_val","Lower_bound","Upper_bound"]]
df_sim_data = pd.read_csv(path_data+'/IL10/data/signaling/IL10_STAT_data.tsv.gz', sep="\t", compression="gzip")
df_EC50_AMP = pd.read_csv(path_data+'/IL10/data/signaling/IL10_EC50_AMP.tsv.gz', sep="\t", compression="gzip")
df_IC_data = pd.read_csv(path_data+'/IL10/data/expression/whole_dataset_IL10.tsv.gz', sep="\t", compression="gzip")

# Get only data on certain variant
variants = ["WT","Super-10","R5A11D"]
df_sim_data = df_sim_data.loc[df_sim_data["Variant"].isin(variants)]
df_EC50_AMP = df_EC50_AMP.loc[df_EC50_AMP["Variant"].isin(variants)]
df_bind = df_bind.loc[df_bind["Variant"].isin(variants)]

# Cells to fit
cells = ["T8.Mean","T4.Mean","MO.Mean","ACH-000146","ACH-000786"]
df_sim_data = df_sim_data.loc[df_sim_data["Cell_type"].isin(cells)]
df_EC50_AMP = df_EC50_AMP.loc[df_EC50_AMP["Cell_type"].isin(cells)]

# Reduce the weight of the initial conditions dataset to run faster simulations]
cells = df_EC50_AMP["Cell_type"].drop_duplicates().values
df_IC_data = df_IC_data.loc[df_IC_data["Cell"].isin(cells)]

# Parameters data
param_depen = {'k_IL_RB_f':['k_IL_RA_RB_f'],'k_IL_RA_RB_f':['k_IL_RB_f'],'k_IL_RB_b':['k_IL_RA_RB_b'],'k_IL_RA_RB_b':['k_IL_RB_b']}
param_unique_var = {"WT": df_bind.columns[1:],'R5A11D':['k_IL_RB_f','k_IL_RB_b','k_IL_RA_RB_f','k_IL_RA_RB_b'],'Super-10':['k_IL_RB_f','k_IL_RB_b','k_IL_RA_RB_f','k_IL_RA_RB_b'],'10-DE':['k_IL_RB_f','k_IL_RB_b','k_IL_RA_RB_f','k_IL_RA_RB_b']}

# Simulation and ABC-SMC variables
num_IL_sim = 6
num_cores_ABC_SMC = 22
ABC_SMC_path = "pyABC_inference_ODE_mut.db"
conf95_interval = 1.25 #1.5 is to much since a lot of the population is out of bounds -> algorithm can't decrease the epsilon
pop_size_init = 100 
pop_size_max = 250
E_cv = 0.275 # Use 0.25 as maximum if the simulations take too long. 0.15 -> 300 indiv, 0.2 -> 100 indiv
initial_epsilon = 0.75
min_epsilon = 5e-5
max_num_iter = 15 # Use 10 for variants where only one parameter needs to be infered
model_function_dict = {
    "IL10_RAp": model_function_IL10_RAp
}

print("Beginning model selection and parameter inference!")
model_selection_ABC_SMC(model_function_dict, df_fit_data, df_bind, df_EC50_AMP, df_IC_data, df_sim_data, param_depen, num_IL_sim, num_cores_ABC_SMC, ABC_SMC_path, conf95_interval, pop_size_init, pop_size_max, E_cv, min_epsilon, max_num_iter, param_unique_var, initial_epsilon)
