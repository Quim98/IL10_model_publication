# Script predicting pSTAT1 and pSTAT3, using as initial conditions the mean expression of each cell type in the immune dicitonary dataset. This will be used to predict genes whose change in expression is correlated with pSTAT

import numpy as np
from src.models.IL10_RAp_ODE import model_function
import pandas as pd
from statistics import median
from multiprocessing import Pool
import matplotlib.pyplot as plt
from src.simulate_func import set_simulation_cell_list,simulate_parallel_ODE
from time import time

# Import datasets
df_IC_data = pd.read_csv('data/expression/immune_dict_dataset_IL10.tsv.gz', sep='\t', compression='gzip')
df_bind = pd.read_csv('data/binding/IL10_data_param_ABC_SMC_IL10_RAp_ODE.csv')

# Create dataset of simulations (PBS only Mean)
cell_list = [cell for cell in list(dict.fromkeys(df_IC_data["Cell"])) if "Mean" in cell]
cell_list = [cell for cell in cell_list if "PBS_" in cell]

# Set up simulations
variant_list = ["WT"]
STAT_list = ["pSTAT3"]
num_sim = 2
low_IL = 1e-11
high_IL = 2.437003460544914e-06
df_sim = set_simulation_cell_list(df_bind, df_IC_data, cell_list, variant_list, STAT_list, num_sim, low_IL, high_IL)

# Simulate
num_cores = 24
df_res = simulate_parallel_ODE(df_sim, num_cores, model_function)
df_res.to_csv("results/immune_dict/simulations_cytk_dict_RAp_ODE.csv")

# Set up simulations (STAT1)
STAT_list = ["pSTAT1"]
df_sim = set_simulation_cell_list(df_bind, df_IC_data, cell_list, variant_list, STAT_list, num_sim, low_IL, high_IL)

# Simulate
df_res = simulate_parallel_ODE(df_sim, num_cores, model_function)

df_res.to_csv("results/immune_dict/simulations_cytk_dict_RAp_ODE_S1.csv")

