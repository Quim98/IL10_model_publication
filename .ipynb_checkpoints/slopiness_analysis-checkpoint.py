# Script performing slopiness analysis to rank the model’s parameters by their effect on its prediction.

import numpy as np
#from src.models.IL10 import model_function
from src.models.IL10_RAp_seq import model_function
#from src.models.IL10_MS1 import model_function
import pandas as pd
from statistics import median
from sys import exit
from scipy.optimize import curve_fit
from multiprocessing import Pool
import matplotlib.pyplot as plt
from src.simulate_func import simulate,update_df_bind_up_down,local_slopiness,calculate_cost_slopiness_fit,set_simulation_slopiness,hill_func,simulate_parallel
from multiprocessing import Pool
from time import time

start_time = time()
# Import datasets
df_IC_data = pd.read_csv('data/expression/whole_dataset_IL10.tsv.gz', sep="\t", compression="gzip")
df_EC50_AMP = pd.read_csv('data/signaling/IL10_EC50_AMP.tsv.gz', sep="\t", compression="gzip")
df_bind = pd.read_csv('data/binding/IL10_data_param.csv')
df_sim_data =  pd.read_csv('data/signaling/IL10_STAT_data.tsv.gz', sep="\t", compression="gzip")
num_IL_sim = 25
num_cores = 24
    
param_depen = {} # We want to see the isolated effect of each parameter
param_mut = [col for col in df_bind.columns[2:-1 ] if "_M_" not in col]
# IL10: [col for col in df_bind.columns[2:-2] if "_f" not in col and "_M_" not in col]
# IL10_RAp: [col for col in df_bind.columns[2:-1 ] if "_M_" not in col]
# IL10_MS1: [col for col in df_bind.columns[2:] if "PHOS" not in col]

# Analyze the model's slopiness
cells_target = cells_notarget = 0
mode = "Fitting"
delta_0 = 5e-2
H = local_slopiness(delta_0, mode, df_bind, df_sim_data, df_IC_data, df_EC50_AMP, num_IL_sim, cells_target, cells_notarget, num_cores, model_function, param_mut, param_depen)
#np.savetxt('results/slopiness/Hessian_nofit_IL10.csv', H, delimiter=',')
np.savetxt('results/slopiness/Hessian_nofit_IL10_RAp.csv', H, delimiter=',')
#np.savetxt('results/slopiness/Hessian_nofit_IL10_MS1.csv', H, delimiter=',')

end_time = time()  # Record the end time
execution_time = end_time - start_time  # Calculate execution time
print(f"Execution Time: {execution_time:.6f} seconds")


   