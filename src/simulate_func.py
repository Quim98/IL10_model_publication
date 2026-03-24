import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import curve_fit
from multiprocessing import Pool
from itertools import combinations_with_replacement
import sympy as sp
from scipy.optimize import fsolve
import pyabc
import pyabc.sge
from scipy.interpolate import interp1d
from time import time

######################################################################################################################################################

def hill_func(inp,B,K):
    """ Hill function to extract EC50s and amplitudes from experimental and model-simulated dose-response curves
    Parameters:
        inp -> IL-10 concentration list
        B -> amplitude
        K -> EC50
    Returns: pSTAT list
    """
    out=B*inp/(K+inp)
    return out
    
######################################################################################################################################################

def simulate_parallel_ODE(df_sim_data, num_cores, model_func):
    """ Simulate ODE models across several conditions in a parallel manner
    Parameters:
        df_sim_data -> Pandas dataframe of parameters and initial conditions (see function set_simulation)
        num_cores -> number of cores for parallel execution
        model_func -> function of the IL-10 model
    Returns: Model simulation stored in df_sim_data["Result"]
    """
    param_parallel = []
    for i in range(0,len(df_sim_data.index)):
        param_parallel.append([df_sim_data.loc[[i]]])
    with Pool(processes=num_cores) as pool:
        out_vec = pool.map(model_func, param_parallel)  
    return pd.concat(out_vec, axis=0)

######################################################################################################################################################

def simulate_parallel(df_sim_data, model_func):
    """ Simulate simplified steady-state solutions across several conditions a parallel manner
    Parameters:
        df_sim_data -> Pandas dataframe of parameters and initial conditions (see function set_simulation)
        model_func -> function of the IL-10 model
    Returns: Model simulation stored in df_sim_data["Result"]
    """
    Na=6.023e23
    Width_PM = 1e-7 # In dm
    # Intial guess for a plot__variant is that all receptors and IL is unbound (IL is at low concentrations)
    x_guess = [10**df_sim_data.loc[[df_sim_data.index[0]],"RA0"].values[0]/2,10**df_sim_data.loc[[df_sim_data.index[0]],"RB0"].values[0]/2,df_sim_data.loc[[df_sim_data.index[0]],"IL0"].values[0]*Na*df_sim_data.loc[[df_sim_data.index[0]],"vol_EC"].values[0]/2,1,df_sim_data.loc[[df_sim_data.index[0]],"M0"].values[0]/2,10**df_sim_data.loc[[df_sim_data.index[0]],"STAT10"].values[0]/2,10**df_sim_data.loc[[df_sim_data.index[0]],"STAT30"].values[0]/2]
    # Save number of plot and variant to check if next simualtion is the first one from a curve
    plot_num_var = df_sim_data.loc[[df_sim_data.index[0]],"Plot"].astype("str").values[0]+"__"+df_sim_data.loc[[df_sim_data.index[0]],"Variant"].values[0]
    # For all IC
    for i in df_sim_data.index:
        # If we are in the first IC or we are in a new plot or new variant in the same plot guess is all IL and receptors unbound
        if plot_num_var != df_sim_data.loc[[i],"Plot"].astype("str").values[0]+"__"+df_sim_data.loc[[i],"Variant"].values[0] or i == df_sim_data.index[0]:
            x_guess = [10**df_sim_data.loc[[i],"RA0"].values[0]/2,10**df_sim_data.loc[[i],"RB0"].values[0]/2,df_sim_data.loc[[i],"IL0"].values[0]*Na*df_sim_data.loc[[i],"vol_EC"].values[0]/2,1,df_sim_data.loc[[i],"M0"].values[0]/2,10**df_sim_data.loc[[i],"STAT10"].values[0]/2,10**df_sim_data.loc[[i],"STAT30"].values[0]/2]
        # If last simulation is in the same plot and with the same variant we use the previous result as IC 
        else:
            x_guess = [x_guess[0],x_guess[1],x_guess[2]*df_sim_data.loc[[i],"IL0"].values[0]/df_sim_data.loc[[i-1],"IL0"].values[0],x_guess[3],x_guess[4],x_guess[5],x_guess[6]]
        # Simulate: get result model and the IC for the next simulation
        PA_res,x_guess = model_func([df_sim_data.loc[[i]],x_guess])
        plot_num_var = df_sim_data.loc[[i],"Plot"].astype("str").values[0]+"__"+df_sim_data.loc[[i],"Variant"].values[0]
        df_sim_data.loc[i,"Result"] = PA_res["Result"].values[0]
    return df_sim_data

######################################################################################################################################################

def simulate(df_sim_data, num_cores, model_func):
    """ Simulate simplified steady-state solutions across several conditions a parallel manner
    Parameters:
        df_sim_data -> Pandas dataframe of parameters and initial conditions (see function set_simulation)
        num_cores -> number of cores for parallel execution
        model_func -> function of the IL-10 model
    Returns: Model simulation stored in df_sim_data["Result"]
    """
    param_parallel = []
    for plot_num,var in df_sim_data[["Plot","Variant"]].drop_duplicates().values:
        param_parallel.append((df_sim_data.loc[(df_sim_data["Plot"]==plot_num)&(df_sim_data["Variant"]==var)],model_func))
    with Pool(processes=num_cores) as pool:
        out_vec = pool.starmap(simulate_parallel, param_parallel)
    return pd.concat(out_vec, axis=0)

######################################################################################################################################################    

def simulate_sequential(df_sim_data, model_func):
    """ Simulate simplified steady-state solutions across several conditions in a sequential manner
    Parameters:
        df_sim_data -> Pandas dataframe of parameters and initial conditions (see function set_simulation)
        model_func -> function of the IL-10 model
    Returns: Model simulation stored in df_sim_data["Result"]
    """
    Na=6.023e23
    Width_PM = 1e-7 # In dm
    # Intial guess for a plot__variant is that all receptors and IL is unbound (IL is at low concentrations)
    x_guess = [10**df_sim_data.loc[[0],"RA0"].values[0]/2,10**df_sim_data.loc[[0],"RB0"].values[0]/2,df_sim_data.loc[[0],"IL0"].values[0]*Na*df_sim_data.loc[[0],"vol_EC"].values[0]/2,1,df_sim_data.loc[[0],"M0"].values[0]/2,10**df_sim_data.loc[[0],"STAT10"].values[0]/2,10**df_sim_data.loc[[0],"STAT30"].values[0]/2]
    # Save number of plot and variant to check if next simualtion is the first one from a curve
    plot_num_var = df_sim_data.loc[[0],"Plot"].astype("str").values[0]+"__"+df_sim_data.loc[[0],"Variant"].values[0]
    # For all IC
    for i in range(0,len(df_sim_data.index)):
        # If we are in the first IC or we are in a new plot or new variant in the same plot guess is all IL and receptors unbound
        if plot_num_var != df_sim_data.loc[[i],"Plot"].astype("str").values[0]+"__"+df_sim_data.loc[[i],"Variant"].values[0] or i == 0:
            x_guess = [10**df_sim_data.loc[[i],"RA0"].values[0]/2,10**df_sim_data.loc[[i],"RB0"].values[0]/2,df_sim_data.loc[[i],"IL0"].values[0]*Na*df_sim_data.loc[[i],"vol_EC"].values[0]/2,1,df_sim_data.loc[[i],"M0"].values[0]/2,10**df_sim_data.loc[[i],"STAT10"].values[0]/2,10**df_sim_data.loc[[i],"STAT30"].values[0]/2]
        # If last simulation is in the same plot and with the same variant we use the previous result as IC 
        else:
            x_guess = [x_guess[0],x_guess[1],x_guess[2]*df_sim_data.loc[[i],"IL0"].values[0]/df_sim_data.loc[[i-1],"IL0"].values[0],x_guess[3],x_guess[4],x_guess[5],x_guess[6]]
        # Simulate: get result model and the IC for the next simulation
        PA_res,x_guess = model_func([df_sim_data.loc[[i]],x_guess])
        plot_num_var = df_sim_data.loc[[i],"Plot"].astype("str").values[0]+"__"+df_sim_data.loc[[i],"Variant"].values[0]
        df_sim_data.loc[i,"Result"] = PA_res["Result"].values[0]
    return df_sim_data

######################################################################################################################################################    

def simulate_sequential_ODE(df_sim_data, model_func):
    """ Simulate simplified steady-state solutions across several conditions in a sequential manner
    Parameters:
        df_sim_data -> Pandas dataframe of parameters and initial conditions (see function set_simulation)
        model_func -> function of the IL-10 model
    Returns: Model simulation stored in df_sim_data["Result"]
    """
    # For all IC
    for i in range(0,len(df_sim_data.index)):
        # Simulate: get result model
        PA_res = model_func([df_sim_data.loc[[i]]])
        df_sim_data.loc[i,"Result"] = PA_res["Result"].values[0]
    return df_sim_data
    
######################################################################################################################################################

def set_simulation(df_bind, df_sim_data, df_IC_data, num_sim):
    """ Create a dataset to simulate dose response curves across a certain number of IL-10 doses.
    Parameters:
        df_bind -> Pandas dataframe of parameter values
        df_sim_data -> Pandas dataframe of cells to simulate 
        df_IC_data -> Pandas dataframe of intial conditions
        num_sim -> Number of IL-10 doses per dose-response curve
    Returns: Pandas dataframe of parameters and initial conditions to simulate the IL-10 model
    """
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","STAT10","STAT30","Result"])
    for plot_num in list(dict.fromkeys(df_sim_data["Plot"])):
        df_plot = df_sim_data.loc[df_sim_data["Plot"] == plot_num]
        IL_vec = 10**(np.linspace(np.log10(df_plot["IL"].min())-1, np.log10(df_plot["IL"].max())+1, num_sim))
        RA0 = df_IC_data.loc[(df_IC_data["Cell"]==df_plot["Cell_type"].values[0])&(df_IC_data["Gene"]=="I10R1_HUMAN"),"Log10 Prot. count"].values[0]
        RB0 = df_IC_data.loc[(df_IC_data["Cell"]==df_plot["Cell_type"].values[0])&(df_IC_data["Gene"]=="I10R2_HUMAN"),"Log10 Prot. count"].values[0]
        STAT10 = df_IC_data.loc[(df_IC_data["Cell"]==df_plot["Cell_type"].values[0])&(df_IC_data["Gene"]=="STAT1_HUMAN"),"Log10 Prot. count"].values[0]
        STAT30 = df_IC_data.loc[(df_IC_data["Cell"]==df_plot["Cell_type"].values[0])&(df_IC_data["Gene"]=="STAT3_HUMAN"),"Log10 Prot. count"].values[0]
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            df_var = df_plot.loc[df_plot["Variant"] == variant]
            for i in range(0,num_sim):
                df_sim.loc[len(df_sim.index)] = df_var.loc[df_var.index[0]].tolist()[:-2] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [IL_vec[i],RA0,RB0,STAT10,STAT30,np.nan]
    return df_sim

######################################################################################################################################################

def set_simulation_cell_list(df_bind, df_IC_data, cell_list, variant_list, STAT_list, num_sim, low_IL, high_IL):
    """ Create a dataset to simulate dose response curves across a certain number of IL-10 doses, STATs and cell types.
    Parameters:
        df_bind -> Pandas dataframe of parameter values
        df_IC_data -> Pandas dataframe of intial conditions
        cell_list -> List of cell types to simulate (Need to be in df_IC_data!)
        variant_list -> List of IL-10 variants to simulate (Need to be in df_bind!)
        STAT_list -> List of STATs to simulate (pSTAT1 or/and pSTAT3)
        num_sim -> Number of IL-10 doses per dose-response curve
        low_IL -> Lowest IL-10 concentration to simulate
        high_IL -> Highest IL-10 concentration to simulate
    Returns: Pandas dataframe of parameters and initial conditions to simulate the IL-10 model
    """
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","STAT10","STAT30","Result"])
    plot_i = 1
    for cell in cell_list:
        IL_vec = 10**(np.linspace(np.log10(low_IL), np.log10(high_IL), num_sim))
        RA0 = df_IC_data.loc[(df_IC_data["Cell"]==cell)&(df_IC_data["Gene"]=="I10R1_HUMAN"),"Log10 Prot. count"].values[0]
        RB0 = df_IC_data.loc[(df_IC_data["Cell"]==cell)&(df_IC_data["Gene"]=="I10R2_HUMAN"),"Log10 Prot. count"].values[0]
        STAT10 = df_IC_data.loc[(df_IC_data["Cell"]==cell)&(df_IC_data["Gene"]=="STAT1_HUMAN"),"Log10 Prot. count"].values[0]
        STAT30 = df_IC_data.loc[(df_IC_data["Cell"]==cell)&(df_IC_data["Gene"]=="STAT3_HUMAN"),"Log10 Prot. count"].values[0]
        for STAT_type in STAT_list:
            for variant in variant_list:
                for i in range(0,num_sim):
                    df_sim.loc[len(df_sim.index)] = [plot_i,cell,variant,STAT_type] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [IL_vec[i],RA0,RB0,STAT10,STAT30,np.nan]
            plot_i += 1
    return df_sim

######################################################################################################################################################

def set_simulation_slopiness(df_bind, df_IC_data, df_EC50_AMP, num_IL_sim):
    """ Create a dataset to simulate dose response curves for the slopiness analysis (IL-10 doses are centred on the EC50s of experimental assays).
    Parameters:
        df_bind -> Pandas dataframe of parameter values
        df_IC_data -> Pandas dataframe of intial conditions
        df_EC50_AMP -> Pandas dataframe of EC50s and ampitudes of experimental data
        num_IL_sim -> Number of IL-10 doses per dose-response curve
    Returns: Pandas dataframe of parameters and initial conditions to simulate the IL-10 model
    """
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","STAT10","STAT30","Result"])
    for plot_num in df_EC50_AMP["Plot"].drop_duplicates():
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        RA0 = df_IC_data.loc[(df_IC_data["Cell"]==df_plot_EC50["Cell_type"].values[0])&(df_IC_data["Gene"]=="I10R1_HUMAN"),"Log10 Prot. count"].values[0]
        RB0 = df_IC_data.loc[(df_IC_data["Cell"]==df_plot_EC50["Cell_type"].values[0])&(df_IC_data["Gene"]=="I10R2_HUMAN"),"Log10 Prot. count"].values[0]
        STAT10 = df_IC_data.loc[(df_IC_data["Cell"]==df_plot_EC50["Cell_type"].values[0])&(df_IC_data["Gene"]=="STAT1_HUMAN"),"Log10 Prot. count"].values[0]
        STAT30 = df_IC_data.loc[(df_IC_data["Cell"]==df_plot_EC50["Cell_type"].values[0])&(df_IC_data["Gene"]=="STAT3_HUMAN"),"Log10 Prot. count"].values[0]
        if np.isnan(df_plot_EC50["EC50"].values[0]):
            IL_vec =  10**(np.linspace(df_EC50_AMP["EC50"].mean()-3, df_EC50_AMP["EC50"].mean()+2, num_IL_sim))
        else:
            IL_vec =  10**(np.linspace(df_plot_EC50["EC50"].mean()-3, df_plot_EC50["EC50"].mean()+2, num_IL_sim))
        for variant in list(dict.fromkeys(df_plot_EC50["Variant"])):
            df_var_EC50 = df_plot_EC50.loc[df_plot_EC50["Variant"] == variant]
            for i in range(0,num_IL_sim):
                df_sim.loc[len(df_sim.index)] = df_var_EC50.loc[df_var_EC50.index[0]].tolist()[:-7] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [IL_vec[i],RA0,RB0,STAT10,STAT30,np.nan]
    return df_sim

######################################################################################################################################################

def update_df_bind_up_down(df_bind, param, param_depen, delta):
    """ Modify model parameters up or down to estimate the derivatives in the Hessian.
    Parameters:
        df_bind -> Pandas dataframe of parameter values
        param -> Parameter to modify
        param_depen -> Parameter dependencies dicitionary
        delta -> How much to modify the parameter value to estimate the derivatives in the Hessian
    Returns: Pandas dataframe of parameter values
    """
    # Create copies of binding dataframe
    df_bind_up = df_bind.copy()
    df_bind_down = df_bind.copy()

    # Update df_bind with new parameter values
    df_bind_up[param] = df_bind[param] + delta*df_bind[param]/2
    df_bind_down[param] = df_bind[param] - delta*df_bind[param]/2

    # Parameter dependancies
    if param in param_depen.keys():
        for param_d in param_depen[param]:
            df_bind_up[param_d] = df_bind[param_d] + delta*df_bind[param_d]/2
            df_bind_down[param_d] = df_bind[param_d] - delta*df_bind[param_d]/2
            
    return df_bind_up, df_bind_down

######################################################################################################################################################

def calculate_cost_slopiness_fit(df_sim, df_EC50_AMP, w_EC50, w_AMP):
    """ Given model simulations on dose-response curves, calculate as a cost function the weighted sum of squares of the error on the IL-10 concentration producing 50% of the maximal response (EC50) and amplitude (AMP)
    Parameters:
        df_sim -> Pandas dataframe of model simulations
        df_EC50_AMP -> Pandas dataframe of EC50s and ampitudes of experimental data
        w_EC50 -> Weight of the EC50 error (calculated from raw parameter values)
        w_AMP -> Weight of the amplitude error (calculated from raw parameter values)
    Returns: Pandas dataframe of EC50s and ampitudes of experimental data and model simulations and their errors
    """
    for plot_num in list(dict.fromkeys(df_sim["Plot"])):
        df_plot = df_sim.loc[df_sim["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        variant = "WT"
        variant_list = df_plot["Variant"].drop_duplicates().to_list()
        variant_list.remove("WT")
        variant_list = ["WT"]+variant_list
        i = 0
        for variant in variant_list:
            if i == 0:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                max_WT = df_var["Result"].max() 
                fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [110, df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[1])
            else:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max()*100/(max_WT)+10, df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[1])
            i += 1
    df_EC50_AMP["EC50_e"] = ((df_EC50_AMP["EC50"] - df_EC50_AMP["EC50_m"])**2/w_EC50).fillna(0)
    df_EC50_AMP["Amp_e"] = ((df_EC50_AMP["Amp"] - df_EC50_AMP["Amp_m"])**2/w_AMP).fillna(0)
    df_EC50_AMP["Error"] = df_EC50_AMP["EC50_e"] + df_EC50_AMP["Amp_e"]
    
    return df_EC50_AMP

######################################################################################################################################################

def local_slopiness(delta_0, df_bind, df_sim_data, df_IC_data, df_EC50_AMP, num_IL_sim, num_cores, model_function, param_mut, param_depen):
    """ Perform a local slopiness analysis by computing the Levenberg-Marquardt Hessian matrix of the model’s cost function 
    Parameters:
        delta_0 -> How much to modify the parameter value to estimate the derivatives in the Hessian
        df_bind -> Pandas dataframe of parameter values
        df_sim_data -> Pandas dataframe of cells to simulate
        df_IC_data -> Pandas dataframe of intial conditions
        df_EC50_AMP -> Pandas dataframe of EC50s and ampitudes of experimental data
        num_IL_sim -> Number of IL-10 doses per dose-response curve
        num_cores -> Number of cores for parallel execution
        model_function -> function of the IL-10 model
        param_mut -> List of parameters to perform the slopiness analysis
        param_depen -> Parameter dependencies dicitionary
    Returns: Levenberg-Marquardt Hessian matrix
    """
    # Perform an initial simualtion to get weights
    # Create dataset of simulations
    df_sim = set_simulation_slopiness(df_bind, df_IC_data, df_EC50_AMP, num_IL_sim)

    # Perform simulations with raw parameters
    df_sim = simulate(df_sim, num_cores, model_function)

    # Calculate EC50 and Amplitude error to add weight
    for plot_num in list(dict.fromkeys(df_sim["Plot"])):
        df_plot = df_sim.loc[df_sim["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        variant = "WT"
        variant_list = df_plot["Variant"].drop_duplicates().to_list()
        variant_list.remove("WT")
        variant_list = ["WT"]+variant_list
        i = 0
        for variant in variant_list:
            if i == 0:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                max_WT = df_var["Result"].max()
                fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [110, df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[1])
            else:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max()*100/(max_WT)+10, df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[1])
            i += 1
        
        # Use same cost function as parameter inference (weighted sum of squares)
        df_EC50_AMP["EC50_e"] = (df_EC50_AMP["EC50"] - df_EC50_AMP["EC50_m"])**2
        df_EC50_AMP["Amp_e"] = (df_EC50_AMP["Amp"] - df_EC50_AMP["Amp_m"])**2
        w_EC50 = df_EC50_AMP["EC50_e"].sum()*1
        w_AMP = df_EC50_AMP["Amp_e"].sum()*5

    df_EC50_AMP.to_csv("first_sim.csv") # BORRAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("EC50 mean distance: " + str(w_EC50/2))
    print("Amplitude mean distance: " + str(w_AMP/2))

    # Local slopiness algorithm begins
    N_param = len(param_mut)
    N_obs = len(df_EC50_AMP.index)
    delta = delta_0
    Jacobian = np.zeros([N_obs,N_param])
    k = 0
    while True:
        k = k + 1
        j = 0
        for param in param_mut:
            # Derive using central differences
            df_bind_up, df_bind_down = update_df_bind_up_down(df_bind, param, param_depen, delta)

            # Create dataset of simulations
            df_sim_up = set_simulation_slopiness(df_bind_up, df_IC_data, df_EC50_AMP, num_IL_sim)
            df_sim_down = set_simulation_slopiness(df_bind_down, df_IC_data, df_EC50_AMP, num_IL_sim)

            # Simulate
            df_res_up = simulate(df_sim_up, num_cores, model_function)
            df_res_down = simulate(df_sim_down, num_cores, model_function)

            # Get residuals
            df_EC50_AMP_up = calculate_cost_slopiness_fit(df_res_up, df_EC50_AMP.copy(), w_EC50, w_AMP)
            df_EC50_AMP_down = calculate_cost_slopiness_fit(df_res_down, df_EC50_AMP.copy(), w_EC50, w_AMP)

            # Derive the cost function using the central differences
            iJ = 0
            for i in df_EC50_AMP_up.index:
                Jacobian[iJ,j] = (df_EC50_AMP_up.loc[i,"Error"] - df_EC50_AMP_down.loc[i,"Error"])/(delta) # Divide cost per number of samples to normalize
                iJ += 1
            j += 1
        
        # Calculate the Hessian using the Gauss-Newton aproximation
        Hessian = np.zeros([N_param,N_param])
        for l in range(0,N_param):
            for m in range(0,N_param):
                for i in range(0,N_obs):
                    Hessian[l,m] = Hessian[l,m] + Jacobian[i,l]*Jacobian[i,m]
        

        # Check that there are no complex, NaNs or inf eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(Hessian)
        if not np.iscomplexobj(eigenvalues)  and not np.all(np.isinf(eigenvalues)) and not np.all(np.isnan(eigenvalues)):
            U, svd_eigenvalues, VT = np.linalg.svd(Hessian)

            # Check for numerical noise
            cond = np.where(abs(svd_eigenvalues/max(svd_eigenvalues)) <= np.sqrt(np.finfo(float).eps))
            svd_eigenvalues[cond] = np.zeros(len(cond))

            if np.all(svd_eigenvalues >= 0) and not np.iscomplexobj(svd_eigenvalues):
                break
            else:
                delta = delta_0*0.1 + round(np.random.uniform(0,1),1)*(delta_0*10-delta_0*0.1)

        if k >= 10:
            print('The Hessian matrix has negative eigenvalues!')
            break

    return Hessian
    

######################################################################################################################################################

def set_simulation_ABC_MS(df_bind, df_IC_data, df_EC50_AMP, num_IL_sim):
    """ Create a dataset to simulate dose response curves for the ABC-SMC model selection/parameter fitting (IL-10 doses are centred on the EC50s of experimental assays).
    Parameters:
        df_bind -> Pandas dataframe of parameter values
        df_IC_data -> Pandas dataframe of intial conditions
        df_EC50_AMP -> Pandas dataframe of EC50s and ampitudes of experimental data
        num_IL_sim -> Number of IL-10 doses per dose-response curve
    Returns: Pandas dataframe of parameters and initial conditions to simulate the IL-10 model
    """
    cell__variant__STAT_list = []
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","STAT10","STAT30","Result"])
    for plot_num in df_EC50_AMP["Plot"].drop_duplicates():
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        RA0 = df_IC_data.loc[(df_IC_data["Cell"]==df_plot_EC50["Cell_type"].values[0])&(df_IC_data["Gene"]=="I10R1_HUMAN"),"Log10 Prot. count"].values[0]
        RB0 = df_IC_data.loc[(df_IC_data["Cell"]==df_plot_EC50["Cell_type"].values[0])&(df_IC_data["Gene"]=="I10R2_HUMAN"),"Log10 Prot. count"].values[0]
        STAT10 = df_IC_data.loc[(df_IC_data["Cell"]==df_plot_EC50["Cell_type"].values[0])&(df_IC_data["Gene"]=="STAT1_HUMAN"),"Log10 Prot. count"].values[0]
        STAT30 = df_IC_data.loc[(df_IC_data["Cell"]==df_plot_EC50["Cell_type"].values[0])&(df_IC_data["Gene"]=="STAT3_HUMAN"),"Log10 Prot. count"].values[0]
        if np.isnan(df_plot_EC50["EC50"].values[0]):
            IL_vec = 10**(np.linspace(df_EC50_AMP["EC50"].mean()-2.5, df_EC50_AMP["EC50"].mean()+2, num_IL_sim))
        else:
            IL_vec =  10**(np.linspace(df_plot_EC50["EC50"].mean()-2.5, df_plot_EC50["EC50"].mean()+2, num_IL_sim))
        variants_plot = list(df_plot_EC50["Variant"].drop_duplicates())
        variants_plot.remove("WT")
        variants_plot = ["WT"]+variants_plot
        for variant in variants_plot:
            cell__variant__STAT = df_plot_EC50["Cell_type"].values[0]+"__"+variant+"__"+df_plot_EC50["STAT_type"].values[0]
            if cell__variant__STAT not in cell__variant__STAT_list:
                df_var_EC50 = df_plot_EC50.loc[df_plot_EC50["Variant"] == variant]
                for i in range(0,len(IL_vec)):
                    df_sim.loc[len(df_sim.index)] = df_var_EC50.loc[df_var_EC50.index[0]].tolist()[:-7] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [IL_vec[i],RA0,RB0,STAT10,STAT30,np.nan]
            cell__variant__STAT_list.append(cell__variant__STAT)
    return df_sim

######################################################################################################################################################

def update_df_sim_bind_ABC_MS(df_sim, param_dict, param_depen, param_unique_var, variants_list):
    """ Update the values of the model parameters in the simulation pandas dataset, following ABC-SMC model selection/parameter fitting.
    Parameters:
        df_sim -> Pandas dataframe of model simulations
        param_dict -> Dictionary of new parameter values
        param_depen -> Parameter dependencies dicitionary
        param_unique_var -> Dictionary of parameters that are exclusive from each IL-10 variant
        variants_list -> List of IL-10 variants to fit parameters
    Returns: Pandas dataframe of model simulations wiht updated parameter values
    """
    # Create copy of binding dataframe
    df_sim_fit = df_sim.copy()
    # Update df_bind with new parameter values
    for variant in variants_list:
        if variant == "WT":
            for param in [key.split('__')[1] for key in param_dict.keys() if key.split('__')[0] == variant]:
                # Copy new parameter to dataframe
                df_sim_fit.loc[df_sim_fit["Variant"] == variant,param] = 10**param_dict[variant+"__"+param]
                # Parameter dependancies
                if param in param_depen.keys():
                    for param_d in param_depen[param]:
                        param1_0 = df_sim.loc[df_sim["Variant"] == variant, param].values[0]
                        param2_0 = df_sim.loc[df_sim["Variant"] == variant, param_d].values[0]
                        df_sim_fit.loc[df_sim_fit["Variant"] == variant, param_d] = (10**param_dict[variant+"__"+param])/param1_0*param2_0
                # Parameter which are the same for WT and other variants must be also changed in the variant parameter row
                for variant_WT_mut in variants_list[1:]: # For all the non-WT variants
                    if param not in param_unique_var[variant_WT_mut]:
                        param_dep_bool = True # If parameter is not unique to the variant, then it must be changed following the WT
                        for param2 in param_unique_var[variant_WT_mut]:
                            if param2 in param_depen.keys(): 
                                if param == param_depen[param2]:
                                    param_dep_bool = False # If parameter is not unique to the variant, but is dependant to a unique one, then it must be changed following the variant parameter
                        # Change the variant non-unqiue parameter and the ones that depend on it, following the WT one
                        if param_dep_bool:
                            df_sim_fit.loc[df_sim_fit["Variant"] == variant_WT_mut, param] = 10**param_dict[variant+"__"+param]
                            # Parameter dependancies
                            if param in param_depen.keys():
                                for param_d in param_depen[param]:
                                    param1_0 = df_sim.loc[df_sim["Variant"] == variant, param].values[0]
                                    param2_0 = df_sim.loc[df_sim["Variant"] == variant, param_d].values[0]
                                    df_sim_fit.loc[df_sim_fit["Variant"] == variant_WT_mut, param_d] = (10**param_dict[variant+"__"+param])/param1_0*param2_0
        # If parameter is not specified as WT, then its unique to the variant -> change only parameter and the ones that depend on it
        else:
            for param in [key.split('__')[1] for key in param_dict.keys() if key.split('__')[0] == variant]:
                df_sim_fit.loc[df_sim_fit["Variant"] == variant, param] = 10**param_dict[variant+"__"+param]
                # Parameter dependancies
                if param in param_depen.keys():
                    for param_d in param_depen[param]:
                        param1_0 = df_sim.loc[df_sim["Variant"] == variant, param].values[0]
                        param2_0 = df_sim.loc[df_sim["Variant"] == variant, param_d].values[0]
                        df_sim_fit.loc[df_sim_fit["Variant"] == variant, param_d] = (10**param_dict[variant+"__"+param])/param1_0*param2_0
    return df_sim_fit

######################################################################################################################################################

def model_selection_ABC_SMC(model_function_dict, df_fit_data, df_bind, df_EC50_AMP, df_IC_data, df_sim_data, param_depen, num_IL_sim, num_cores_ABC_SMC, ABC_SMC_path, conf95, pop_size_init, pop_size_max, E_cv, min_epsilon, max_num_iter, param_unique_var, init_epsilon):
    """ Perform ABC-SMC model selection/parameter fitting.
    Parameters:
        model_function_dict -> Dictionary of model functions
        df_fit_data -> Pandas dataframe of parameters to fit
        df_bind -> Pandas dataframe of parameter values
        df_EC50_AMP -> Pandas dataframe of EC50s and ampitudes of experimental data
        df_IC_data -> Pandas dataframe of intial conditions
        df_sim_data -> Pandas dataframe of cells to simulate
        param_depen -> Parameter dependencies dicitionary
        num_IL_sim -> Number of IL-10 doses per dose-response curve
        num_cores_ABC_SMC -> Number of cores for parallel execution
        ABC_SMC_path -> path where to store the ABC-SMC results
        conf95 -> Width of the prior distributions
        pop_size_init -> Size of population at t=0
        pop_size_max -> Maximum size of population
        E_cv -> Error criterion. A smaller value leads generally to larger populations
        min_epsilon -> Error threshold where to stop
        max_num_iter -> Maxmimum number of ABC-SMC iterations
        param_unique_var -> Dictionary of parameters that are exclusive from each IL-10 variant
        init_epsilon -> Initial error threshold
    Returns: None
    """
    # From EC50_AMP_model and EC50_AMP_exp
    def cost_ABC(EC50_AMPm, EC50_AMP0):
        # Get wighted sum of squares (Euclidian distance)    
        df_weight = ((pd.DataFrame.from_dict(EC50_AMP0)-pd.DataFrame.from_dict(EC50_AMPm))**2).transpose()
        e_AMP = df_weight[0].sum()/(w_AMP)
        e_EC50 = df_weight[1].sum()/(w_EC50)
        return e_AMP*e_EC50
     # Create the model's functions
    def create_function(name, model_function):
        # Define the function dynamically. From parameter_dict, simulate the desired model for all cells/variants we need
        def dynamic_model(parameter_dict):
            # If sampled parameters are outside of the bounds do not simulate and output the simulation with original parameter values (high cost -> individual is not selected)
            if False in [(np.log10(df_fit_data.loc[df_fit_data["Var__Param"]==var__param,"Lower_bound"]).values[0] < parameter_dict[var__param]) & (np.log10(df_fit_data.loc[df_fit_data["Var__Param"]==var__param,"Upper_bound"]) > parameter_dict[var__param]).values[0] for var__param in parameter_dict.keys()]:
                EC50_AMP = EC50_AMP_prior.copy()
            
            # If the parameter values are inside the bounds simulate the model
            else:
                # Create new dataset of simulations with updated parameter values
                df_sim = update_df_sim_bind_ABC_MS(df_sim_copy, parameter_dict, param_depen, param_unique_var, variants_fit)
                
                # Simulate
                df_sim = simulate_sequential_ODE(df_sim, model_function)
                df_sim = df_sim.loc[df_sim["Result"]>0]

                # Add missing simulation data on certain plots where the pair of cell__variant is found more than one time but it can be simualted only once
                for plot_num_copy in cell__variant__STAT_missing_in_plot.keys():
                    for plot_num_paste in cell__variant__STAT_missing_in_plot[plot_num_copy]:
                        df_copy = df_sim.loc[df_sim["Plot"]==plot_num_copy].loc[df_sim.loc[df_sim["Plot"]==plot_num_copy]["Variant"]=="WT"]
                        df_copy['Plot'] = plot_num_paste
                        df_sim = pd.concat([df_sim, df_copy], ignore_index=True)
                
                if True in np.isnan(df_sim["Result"].replace([np.inf, -np.inf], np.nan).values):
                    print("Nan or inf simulation results: "+str(df_sim.loc[np.isnan(df_sim["Result"].replace([np.inf, -np.inf], np.nan).values)]))
                if True in np.isnan(df_sim["IL0"].replace([np.inf, -np.inf], np.nan).values):
                    print("Nan or inf simulation IL0: "+str(df_sim.loc[np.isnan(df_sim["IL0"].replace([np.inf, -np.inf], np.nan).values)]))
                
                # Calculate EC50 and Amplitude error to add weight
                EC50_AMP = {key: [] for key in df_EC50_AMP["Plot"].astype(str)+"__"+df_EC50_AMP["Variant"]}
                
                plot_num_old = 0
                # For each celltype/variant simulated
                for key in EC50_AMP.keys():
                    plot_num, variant = key.split("__")
                    plot_num = int(plot_num)
                    df_var = df_sim.loc[(df_sim["Plot"] == plot_num)&(df_sim["Variant"] == variant)]
                    # Data is normalized to the first variant that appears in a plot (Determined from df_sim to be WT)
                    if plot_num_old != plot_num:
                        max_WT = df_var["Result"].max()
                        fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [100, df_var["IL0"].max()*10]))
                        EC50_AMP[key].append(fit[0])
                        EC50_AMP[key].append(np.log10(fit[1]))
                    else:
                        fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max()*100/(max_WT)+5, df_var["IL0"].max()*10]))
                        EC50_AMP[key].append(fit[0])
                        EC50_AMP[key].append(np.log10(fit[1]))
                    plot_num_old = plot_num
            return EC50_AMP
        # Set the function name
        dynamic_model.__name__ = name
    
        return dynamic_model

    # Create the different models' functions dynamically
    model_function_list = [create_function(name, value) for name, value in model_function_dict.items()]
    
    # Save parameters to fit in a way that can be used by pyABC
    param_fit = {} # Dictionary of lists of parameters to fit(variant__parameter)
    for model_name in model_function_dict.keys():
        param_fit[model_name] = list(df_fit_data.loc[df_fit_data["Model"]==model_name,"Var__Param"])
    variants_fit = df_bind["Variant"].values # List of variants to fit
    
    # Get the EC50 and Amplitude of data to fit in a dictionary (pyABC required)
    EC50_AMP_0 = {key: [] for key in df_EC50_AMP["Plot"].astype(str)+"__"+df_EC50_AMP["Variant"]}
    for key in EC50_AMP_0.keys():
        plot_num, variant = key.split("__")
        plot_num = int(plot_num)
        EC50_AMP_0[key].append(df_EC50_AMP.loc[(df_EC50_AMP["Plot"]==plot_num)&(df_EC50_AMP["Variant"]==variant),"Amp"].values[0])
        EC50_AMP_0[key].append(df_EC50_AMP.loc[(df_EC50_AMP["Plot"]==plot_num)&(df_EC50_AMP["Variant"]==variant),"EC50"].values[0])
        
    # Generate the priors
    prior_list = []
    for model_name in model_function_dict.keys():
        prior_dict = {}
        for param in param_fit[model_name]:
            prior_dict[param] = pyabc.RV("norm", loc=np.log10(df_fit_data.loc[df_fit_data["Var__Param"]==param,"Initial_val"].values[0]), scale=conf95/4)
        prior_list.append(pyabc.Distribution(**prior_dict))
    
    # Simulate at the peak of the prior distribution to get cost function weights using "Control" model
    df_bind_init = df_bind.copy()
    
    # Create dataset of simulations
    df_sim = set_simulation_ABC_MS(df_bind_init, df_IC_data, df_EC50_AMP, num_IL_sim)
    for param in set([var__param.split("__")[1] for var__param in df_fit_data["Var__Param"]]):
        df_sim[param] = df_sim[param].astype(float)
    df_sim_copy = df_sim.copy() # Copy simulation dataset that will be used as a template for the rest of parameter values

    # Get plot numbers of repeated pairs of cell__variant_STAT. Keys on dictionary are the plot numbers where WT data has to be copied from. Values on dictionary are the plot numbers where WT data has to be copied to.
    cell__variant__STAT_missing_in_plot = {df_EC50_AMP.loc[(df_EC50_AMP["Cell_type"]+"__"+df_EC50_AMP["Variant"]+"__"+df_EC50_AMP["STAT_type"]).str.contains(cell__variant_STAT),"Plot"].to_list()[0]:df_EC50_AMP.loc[(df_EC50_AMP["Cell_type"]+"__"+df_EC50_AMP["Variant"]+"__"+df_EC50_AMP["STAT_type"]).str.contains(cell__variant_STAT),"Plot"].to_list()[1:] for cell__variant_STAT in (df_EC50_AMP["Cell_type"]+"__"+df_EC50_AMP["Variant"]+"__"+df_EC50_AMP["STAT_type"]).drop_duplicates() if len(df_EC50_AMP.loc[(df_EC50_AMP["Cell_type"]+"__"+df_EC50_AMP["Variant"]+"__"+df_EC50_AMP["STAT_type"]).str.contains(cell__variant_STAT)]) > 1}
    
    # Simulate
    start_time = time()
    df_sim = simulate_sequential_ODE(df_sim, model_function_dict[list(model_function_dict.keys())[0]])
    # Add missing simulation data on certain plots where the pair of cell__variant is found more than one time but it can be simualted only once
    for plot_num_copy in cell__variant__STAT_missing_in_plot.keys():
        for plot_num_paste in cell__variant__STAT_missing_in_plot[plot_num_copy]:
            df_copy = df_sim.loc[df_sim["Plot"]==plot_num_copy].loc[df_sim.loc[df_sim["Plot"]==plot_num_copy]["Variant"]=="WT"]
            df_copy['Plot'] = plot_num_paste
            df_sim = pd.concat([df_sim, df_copy], ignore_index=True)
    end_time = time()  # Record the end time
    execution_time = end_time - start_time  # Calculate execution time
    print(f"Execution time simulation: {execution_time:.6f} seconds")
    
    # Calculate EC50 and Amplitude error to add weight
    EC50_AMP = {key: [] for key in df_EC50_AMP["Plot"].astype(str)+"__"+df_EC50_AMP["Variant"]}
    plot_num_old = 0
    # For each celltype/variant simulated
    for key in EC50_AMP.keys():
        plot_num, variant = key.split("__")
        plot_num = int(plot_num)
        df_var = df_sim.loc[(df_sim["Plot"] == plot_num)&(df_sim["Variant"] == variant)]
        # Data is normalized to the first variant that appears in a plot
        if plot_num_old != plot_num:
            max_WT = df_var["Result"].max()
            fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [100, df_var["IL0"].max()*10]))
            EC50_AMP[key].append(fit[0])
            EC50_AMP[key].append(np.log10(fit[1]))
        else:
            fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max()*100/(max_WT)+5, df_var["IL0"].max()*10]))
            EC50_AMP[key].append(fit[0])
            EC50_AMP[key].append(np.log10(fit[1]))
        plot_num_old = plot_num

    # Get weights for the sum of squares (Euclidian distance function)
    df_weight = ((pd.DataFrame.from_dict(EC50_AMP_0)-pd.DataFrame.from_dict(EC50_AMP))**2).transpose()
    w_AMP = df_weight[0].sum()
    w_EC50 = df_weight[1].sum()
    
    # Save prior EC50_AMP so it can be used to create the bounds
    EC50_AMP_prior = EC50_AMP.copy()

    print("EC50 mean error: " + str((w_EC50/len(EC50_AMP_0.keys()))**(1/2)))
    print("Amplitude mean error: " + str((w_AMP/len(EC50_AMP_0.keys()))**(1/2)))
    print("Inital cost: " + str(cost_ABC(EC50_AMP, EC50_AMP_0)))
    print("Models to select from: "+str(len(model_function_list)))
    
    # Execute ABC-SMC model selection
    abc = pyabc.ABCSMC(model_function_list, prior_list, cost_ABC, sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=num_cores_ABC_SMC), population_size=pyabc.populationstrategy.AdaptivePopulationSize(start_nr_particles=pop_size_init, mean_cv=E_cv, max_population_size=pop_size_max), eps=pyabc.epsilon.MedianEpsilon(initial_epsilon=init_epsilon))
    
    # Get path to save ABC-SMC runs
    abc.new("sqlite:///" + ABC_SMC_path, EC50_AMP_0)

    # Execute ABC-SMC parameter inference
    abc.run(minimum_epsilon=min_epsilon, max_nr_populations=max_num_iter)
    
######################################################################################################################################################