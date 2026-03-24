from pysb import Model, Parameter, Compartment, Monomer, Parameter, Rule, Initial, Observable
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import cython

def model_function(PA):
    """ Model function to simulate the "Receptor Memory" IL-10 signaling model at a specific IL-10 dose, cell type and IL-10 monomeric variant using the full ODE system.
    Parameters:
        PA -> Row of a pandas dataframe containing the values of parameters and initial conditions
    Returns: Row of a pandas dataframe containing the values of parameters and initial conditions + pSTAT1/pSTAT3 at the steady state
    """
    def func_time(IL10):
        if IL10 > 1e-7:
            return 0.1*5,1e4
        else:
            IL10 = abs(np.log10(IL10)) - 3
            return 10**(IL10)/(1e4)*5,10**(IL10)

    PA = PA[0]
    Na=6.023e23
    Width_PM = 1e-7 # In dm
    # Initialize model
    Model()
    
    # Compartment
    Compartment(name='EC', parent=None, dimension=3, size=None)
    Compartment(name='Cell_PM', parent=EC, dimension=2, size=None)
    Compartment(name='Cell_CP', parent=Cell_PM, dimension=3, size=None)
    
    # Monomers
    Monomer('IL', ['bA1', 'bB1'])
    Monomer('RA', ['bA','kA'], {'kA': ['P','U']})
    Monomer('RB', ['bB'])
    
    # Parameters (We put whatever number)   
    Parameter('K_IL_RA_f', PA["k_IL_RA_f"].values[0]/(Na*PA["vol_EC"].values[0]))
    Parameter('K_IL_RA_b', PA["k_IL_RA_b"].values[0])
    Parameter('K_IL_RB_f', PA["k_IL_RB_f"].values[0]/(Na*PA["vol_EC"].values[0]))
    Parameter('K_IL_RB_b', PA["k_IL_RB_b"].values[0])
    
    Parameter('K_IL_RA2_f', PA["k_IL_RA_f"].values[0]/(Na*PA["surf_cell"].values[0]*Width_PM))
    Parameter('K_IL_RA2_b', PA["k_IL_RA_b"].values[0])
    Parameter('K_IL_RA_RB_f', PA["k_IL_RA_RB_f"].values[0]/(Na*PA["surf_cell"].values[0]*Width_PM))
    Parameter('K_IL_RA_RB_b', PA["k_IL_RA_RB_b"].values[0])
    
    Parameter('K_DEPHOS_R', PA["k_DEPHOS_R"].values[0])
    
    # Complexes and reactions (binding of receptors)
    IL_free = IL(bA1=None, bB1=None)
    
    # Binding of first receptor
    IL_RA1_alone = IL(bA1=1, bB1=None) % RA(bA=1)
    Rule('IL_RA1_alone_binding', IL_free + RA(bA=None)| IL_RA1_alone, *[K_IL_RA_f, K_IL_RA_b])
    IL_RB1_alone = IL(bA1=None, bB1=1) % RB(bB=1)
    Rule('IL_RB1_binding_alone', IL_free + RB(bB=None) | IL_RB1_alone, *[K_IL_RB_f, K_IL_RB_b])
    
    # Binding of second receptor  
    IL_RA1p_RB1 = IL(bA1=1, bB1=2) % RA(bA=1,kA='P') % RB(bB=2)
    Rule('IL_RA1_wRB1_binding', IL_RB1_alone + RA(bA=None,kA='U') >> IL_RA1p_RB1, K_IL_RA2_f)
    Rule('IL_RA1p_wRB1_binding', IL_RB1_alone + RA(bA=None,kA='P') | IL_RA1p_RB1, *[K_IL_RA2_f, K_IL_RA2_b])
    
    IL_RA1u_alone = IL(bA1=1, bB1=None) % RA(bA=1,kA='U')
    Rule('IL_wRA1_RB1_binding_DB', IL_RA1u_alone + RB(bB=None) >> IL_RA1p_RB1, K_IL_RA_RB_f)
    IL_RA1p_alone = IL(bA1=1, bB1=None) % RA(bA=1,kA='P')
    Rule('IL_wRA1p_RB1_binding_DB', IL_RA1p_alone + RB(bB=None) | IL_RA1p_RB1, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
    
    # Dephosphorylation of receptors
    Rule('RA_DEPHOS', RA(bA=None, kA='P') >> RA(bA=None, kA='U'), K_DEPHOS_R)
    Rule('IL_RA1_DEPHOS', IL(bA1=1,bB1=None) % RA(bA=1, kA='P') >> IL(bA1=1,bB1=None) % RA(bA=1, kA='U'), K_DEPHOS_R)
    
    # Initial conditions
    Parameter('ILD_0', PA["IL0"].values[0]*Na*PA["vol_EC"].values[0])
    Initial(IL(bA1=None, bB1=None) ** EC, ILD_0)
    
    Parameter('RA_0',10**PA["RA0"].values[0])
    Parameter('RB_0',10**PA["RB0"].values[0])
    Initial(RA(bA=None, kA='U') ** Cell_PM, RA_0)
    Initial(RB(bB=None) ** Cell_PM, RB_0)
    
    # Simulation
    Observable('IL_free', IL(bA1=None, bB1=None) ** EC)
    
    Observable('RA_free', RA(bA=None, kA='U') ** Cell_PM)
    Observable('RAp_free', RA(bA=None, kA='P') ** Cell_PM)
    Observable('RB_free', RB(bB=None) ** Cell_PM)
    
    Observable('IL_RA1', IL(bA1=1, bB1=None) ** EC % RA(bA=1, kA='U') ** Cell_PM)
    Observable('IL_RA1p', IL(bA1=1, bB1=None) ** EC % RA(bA=1, kA='P') ** Cell_PM)
    Observable('IL_RB1', IL(bA1=None, bB1=1) ** EC % RB(bB=1) ** Cell_PM)
    
    Observable('IL_RA1p_RB1', IL(bA1=1, bB1=2) ** EC % RA(bA=1, kA='P') ** Cell_PM % RB(bB=2) ** Cell_PM)
    Observable('RAp', RA(kA='P') ** Cell_PM)
    
    dT, Tf = func_time(PA["IL0"].values[0])
    t = np.arange(0,Tf*1,dT/1000)
    simulator = ScipyOdeSimulator(model, tspan=t, compiler='python', integrator = 'lsoda').run().all

    if PA["STAT_type"].values[0] == "pSTAT3":
        PA["Result"] = simulator['RAp'][-1]
    elif PA["STAT_type"].values[0] == "pSTAT1":
        PA["Result"] = simulator['IL_RA1p_RB1'][-1]
        
    return PA