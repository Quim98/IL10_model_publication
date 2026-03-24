from pysb import Model, Parameter, Compartment, Monomer, Parameter, Rule, Initial, Observable
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import cython

def model_function(PA):
    """ Model function to simulate the "Control" IL-10 signaling model at a specific IL-10 dose, cell type and IL-10 variant using the full ODE system.
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
    Monomer('ILM')
    Monomer('IL', ['bA1', 'bA2', 'bB1', 'bB2','B_t'], {'B_t': ['QB','TB','DB','SB','U']})
    Monomer('RA', ['bA'])
    Monomer('RB', ['bB'])
    
    # Parameters (We put whatever number)
    Parameter('K_ILM_f', PA["k_ILM_f"].values[0]/(Na*PA["vol_EC"].values[0])*2)
    Parameter('K_ILM_b', PA["k_ILM_b"].values[0])
    
    Parameter('K_IL_RA_f', PA["k_IL_RA_f"].values[0]/(Na*PA["vol_EC"].values[0]))
    Parameter('K_IL_RA_b', PA["k_IL_RA_b"].values[0])
    Parameter('K_IL_RB_f', PA["k_IL_RB_f"].values[0]/(Na*PA["vol_EC"].values[0]))
    Parameter('K_IL_RB_b', PA["k_IL_RB_b"].values[0])
    
    Parameter('K_IL_RA2_f', PA["k_IL_RA_f"].values[0]/(Na*PA["surf_cell"].values[0]*Width_PM))
    Parameter('K_IL_RA2_b', PA["k_IL_RA_b"].values[0])
    Parameter('K_IL_RB2_f', PA["k_IL_RB_f"].values[0]/(Na*PA["surf_cell"].values[0]*Width_PM))
    Parameter('K_IL_RB2_b', PA["k_IL_RB_b"].values[0])
    Parameter('K_IL_RA_RB_f', PA["k_IL_RA_RB_f"].values[0]/(Na*PA["surf_cell"].values[0]*Width_PM))
    Parameter('K_IL_RA_RB_b', PA["k_IL_RA_RB_b"].values[0])
    
    # Complexes and reactions (binding of receptors)
    IL_free = IL(bA1=None, bA2=None, bB1=None, bB2=None, B_t='U')
    ILM_free = ILM()
    Rule('IL_binding', ILM_free + ILM_free | IL_free, *[K_ILM_f, K_ILM_b])
    
    IL_RA1_alone = IL(bA1=1, bA2=None, bB1=None, bB2=None, B_t='SB') % RA(bA=1)
    Rule('IL_RA1_alone_binding', IL_free + RA(bA=None)| IL_RA1_alone, *[K_IL_RA_f, K_IL_RA_b])
    IL_RA2_alone = IL(bA1=None, bA2=1, bB1=None, bB2=None, B_t='SB') % RA(bA=1)
    Rule('IL_RA2_alone_binding', IL_free + RA(bA=None) | IL_RA2_alone, *[K_IL_RA_f, K_IL_RA_b])
    IL_RB1_alone = IL(bA1=None, bA2=None, bB1=1, bB2=None, B_t='SB') % RB(bB=1)
    Rule('IL_RB1_binding_alone', IL_free + RB(bB=None) | IL_RB1_alone, *[K_IL_RB_f, K_IL_RB_b])
    IL_RB2_alone = IL(bA1=None, bA2=None, bB1=None, bB2=1, B_t='SB') % RB(bB=1)
    Rule('IL_RB2_binding_alone', IL_free + RB(bB=None) | IL_RB2_alone, *[K_IL_RB_f, K_IL_RB_b])
    
    IL_noRA1_SB = IL(bA1=None, B_t='SB')
    IL_RA1_DB = IL(bA1=1, B_t='DB') % RA(bA=1)
    Rule('IL_RA1_binding_DB', IL_noRA1_SB + RA(bA=None) | IL_RA1_DB, *[K_IL_RA2_f, K_IL_RA2_b])
    IL_noRA2_SB = IL(bA2=None, B_t='SB')
    IL_RA2_DB = IL(bA2=2, B_t='DB') % RA(bA=2)
    Rule('IL_RA2_binding_DB', IL_noRA2_SB + RA(bA=None) | IL_RA2_DB, *[K_IL_RA2_f, K_IL_RA2_b])
    
    IL_noRA1_noRB1_SB = IL(bA1=None, bB1=None, B_t='SB')
    IL_noRA1_RB1_DB = IL(bA1=None, bB1=3, B_t='DB') % RB(bB=3)
    Rule('IL_noRA1_RB1_binding_DB', IL_noRA1_noRB1_SB + RB(bB=None) | IL_noRA1_RB1_DB, *[K_IL_RB2_f, K_IL_RB2_b])
    IL_noRA2_noRB2_SB = IL(bA2=None, bB2=None, B_t='SB')
    IL_noRA2_RB2_DB = IL(bA2=None, bB2=4, B_t='DB') % RB(bB=4)
    Rule('IL_noRA2_RB2_binding_DB', IL_noRA2_noRB2_SB + RB(bB=None) | IL_noRA2_RB2_DB, *[K_IL_RB2_f, K_IL_RB2_b])

    IL_RA1_noRB1_SB = IL(bA1=1, bB1=None, B_t='SB') % RA(bA=1)
    IL_RA1_RB1_DB = IL(bA1=1, bB1=2, B_t='DB') % RA(bA=1) % RB(bB=2)
    Rule('IL_RA1_RB1_binding_DB', IL_RA1_noRB1_SB + RB(bB=None) | IL_RA1_RB1_DB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
    IL_RA2_noRB2_SB = IL(bA2=1, bB2=None, B_t='SB') % RA(bA=1)
    IL_RA2_RB2_DB = IL(bA2=1, bB2=2, B_t='DB') % RA(bA=1) % RB(bB=2)
    Rule('IL_RA2_RB2_binding_DB', IL_RA2_noRB2_SB + RB(bB=None) | IL_RA2_RB2_DB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])

    IL_noRA1_DB = IL(bA1=None, B_t='DB')
    IL_RA1_TB = IL(bA1=1, B_t='TB') % RA(bA=1)
    Rule('IL_RA1_binding_TB', IL_noRA1_DB + RA(bA=None) | IL_RA1_TB, *[K_IL_RA2_f, K_IL_RA2_b])
    IL_noRA2_DB = IL(bA2=None, B_t='DB')
    IL_RA2_TB = IL(bA2=2, B_t='TB') % RA(bA=2)
    Rule('IL_RA2_binding_TB', IL_noRA2_DB + RA(bA=None) | IL_RA2_TB, *[K_IL_RA2_f, K_IL_RA2_b])
    
    IL_RA1_noRB1_DB = IL(bA1=1, bB1=None, B_t='DB') % RA(bA=1)
    IL_RA1_RB1_TB = IL(bA1=1, bB1=3, B_t='TB') % RA(bA=1) % RB(bB=3)
    Rule('IL_RA1_RB1_binding_TB', IL_RA1_noRB1_DB + RB(bB=None) | IL_RA1_RB1_TB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
    IL_noRA1_noRB1_DB = IL(bA1=None, bB1=None, B_t='DB')
    IL_noRA1_RB1_TB = IL(bA1=None, bB1=3, B_t='TB') % RB(bB=3)
    Rule('IL_noRA1_RB1_binding_TB', IL_noRA1_noRB1_DB + RB(bB=None) | IL_noRA1_RB1_TB, *[K_IL_RB2_f, K_IL_RB2_b])
    
    IL_RA2_noRB2_DB = IL(bA2=2, bB2=None, B_t='DB') % RA(bA=2)
    IL_RA2_RB2_TB = IL(bA2=2, bB2=4, B_t='TB') % RA(bA=2) % RB(bB=4)
    Rule('IL_RA2_RB2_binding_TB', IL_RA2_noRB2_DB + RB(bB=None) | IL_RA2_RB2_TB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
    IL_noRA2_noRB2_DB = IL(bA2=None, bB2=None, B_t='DB')
    IL_noRA2_RB2_TB = IL(bA2=None, bB2=4, B_t='TB') % RB(bB=4)
    Rule('IL_noRA2_RB2_binding_TB', IL_noRA2_noRB2_DB + RB(bB=None) | IL_noRA2_RB2_TB, *[K_IL_RB2_f, K_IL_RB2_b])

    IL_noRA1_TB = IL(bA1=None, bA2=2, bB1=3, bB2=4, B_t='TB') % RA(bA=2) % RB(bB=3) % RB(bB=4)
    IL_RA1_RA2_RB1_RB2 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1) % RA(bA=2) % RB(bB=3) % RB(bB=4)
    Rule('IL_RA1_binding_fin', IL_noRA1_TB + RA(bA=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA2_f, K_IL_RA2_b])
    IL_noRA2_TB = IL(bA1=1, bA2=None, bB1=3, bB2=4, B_t='TB') % RA(bA=1) % RB(bB=3) % RB(bB=4)
    Rule('IL_RA2_binding_fin', IL_noRA2_TB + RA(bA=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA2_f, K_IL_RA2_b])
    
    IL_RA1_noRB1_TB = IL(bA1=1, bA2=2, bB1=None, bB2=4, B_t='TB') % RA(bA=1) % RA(bA=2) % RB(bB=4)
    Rule('IL_RA1_RB1_binding_fin', IL_RA1_noRB1_TB+ RB(bB=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
    IL_RA2_noRB2_TB = IL(bA1=1, bA2=2, bB1=3, bB2=None, B_t='TB') % RA(bA=1) % RA(bA=2) % RB(bB=3)
    Rule('IL_RA2_RB2_binding_fin', IL_RA2_noRB2_TB + RB(bB=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
    
    # Initial conditions
    Kd_IL = PA["k_ILM_b"].values[0]/(PA["k_ILM_f"].values[0]/(Na*PA["vol_EC"].values[0]))
    ILM_eq = (-Kd_IL/2 + (Kd_IL**2/4 + 2*Kd_IL*PA["IL0"].values[0]*Na*PA["vol_EC"].values[0])**(1/2))/2
    ILD_eq = (PA["IL0"].values[0]*Na*PA["vol_EC"].values[0]-ILM_eq)/2
    Parameter('ILM_0', ILM_eq)
    Initial(ILM() ** EC , ILM_0)
    Parameter('ILD_0', ILD_eq) #PA["IL0"]*Na*PA["vol_EC"])
    Initial(IL(bA1=None, bA2=None, bB1=None, bB2=None, B_t='U') ** EC, ILD_0)
    
    Parameter('RA_0',10**PA["RA0"].values[0])
    Parameter('RB_0',10**PA["RB0"].values[0])
    Initial(RA(bA=None) ** Cell_PM, RA_0)
    Initial(RB(bB=None) ** Cell_PM, RB_0)
    
    # Simulation
    Observable('ILM_free', ILM() ** EC)
    Observable('IL_free', IL(bA1=None, bA2=None, bB1=None, bB2=None, B_t='U') ** EC)
    Observable('RA_free', RA(bA=None) ** Cell_PM)
    Observable('RB_free', RB(bB=None) ** Cell_PM)
    
    Observable('IL_RA1', IL(bA1=1, bA2=None, bB1=None, bB2=None, B_t='SB') ** EC % RA(bA=1) ** Cell_PM)
    Observable('IL_RA2', IL(bA1=None, bA2=1, bB1=None, bB2=None, B_t='SB') ** EC % RA(bA=1) ** Cell_PM)
    Observable('IL_RB1', IL(bA1=None, bA2=None, bB1=1, bB2=None, B_t='SB') ** EC % RB(bB=1) ** Cell_PM)
    Observable('IL_RB2', IL(bA1=None, bA2=None, bB1=None, bB2=1, B_t='SB') ** EC % RB(bB=1) ** Cell_PM)
    
    Observable('IL_RA1_RA2', IL(bA1=1, bA2=2, bB1=None, bB2=None, B_t='DB') ** EC % RA(bA=1) ** Cell_PM % RA(bA=2) ** Cell_PM)
    Observable('IL_RA1_RB1', IL(bA1=1, bA2=None, bB1=2, bB2=None, B_t='DB') ** EC % RA(bA=1) ** Cell_PM % RB(bB=2) ** Cell_PM)
    Observable('IL_RA1_RB2', IL(bA1=1, bA2=None, bB1=None, bB2=2, B_t='DB') ** EC % RA(bA=1) ** Cell_PM % RB(bB=2) ** Cell_PM)
    Observable('IL_RA2_RB1', IL(bA1=None, bA2=1, bB1=2, bB2=None, B_t='DB') ** EC % RA(bA=1) ** Cell_PM % RB(bB=2) ** Cell_PM)
    Observable('IL_RA2_RB2', IL(bA1=None, bA2=1, bB1=None, bB2=2, B_t='DB') ** EC % RA(bA=1) ** Cell_PM % RB(bB=2) ** Cell_PM)
    Observable('IL_RB1_RB2', IL(bA1=None, bA2=None, bB1=1, bB2=2, B_t='DB') ** EC % RB(bB=1) ** Cell_PM % RB(bB=2) ** Cell_PM)

    Observable('IL_RA1_RA2_RB1', IL(bA1=1, bA2=2, bB1=3, bB2=None, B_t='TB') ** EC % RA(bA=1) ** Cell_PM % RA(bA=2) ** Cell_PM % RB(bB=3) ** Cell_PM)
    Observable('IL_RA1_RA2_RB2', IL(bA1=1, bA2=2, bB1=None, bB2=3, B_t='TB') ** EC % RA(bA=1) ** Cell_PM % RA(bA=2) ** Cell_PM % RB(bB=3) ** Cell_PM)
    Observable('IL_RA1_RB1_RB2', IL(bA1=1, bA2=None, bB1=2, bB2=3, B_t='TB') ** EC % RA(bA=1) ** Cell_PM % RB(bB=2) ** Cell_PM % RB(bB=3) ** Cell_PM)
    Observable('IL_RA2_RB1_RB2', IL(bA1=None, bA2=1, bB1=2, bB2=3, B_t='TB') ** EC % RA(bA=1) ** Cell_PM % RB(bB=2) ** Cell_PM % RB(bB=3) ** Cell_PM)

    Observable('IL_RA1_RA2_RB1_RB2', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1) ** Cell_PM % RA(bA=2) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM)
    
    dT, Tf = func_time(PA["IL0"].values[0])
    t = np.arange(0,Tf/1,dT/50)
    simulator = ScipyOdeSimulator(model, tspan=t, compiler='python', integrator = 'lsoda').run().all

    if PA["STAT_type"].values[0] == "pSTAT3":
        C0 = simulator['IL_RA1_RB1'][-1]+simulator['IL_RA2_RB2'][-1]+simulator['IL_RA1_RA2_RB1'][-1]+simulator['IL_RA1_RA2_RB2'][-1]+simulator['IL_RA1_RB1_RB2'][-1]+simulator['IL_RA2_RB1_RB2'][-1]+2*simulator['IL_RA1_RA2_RB1_RB2'][-1]
        KbS = PA["k_STAT3_f"].values[0]/(Na*PA["vol_cell"].values[0])
        KuS = PA["k_STAT3_b"].values[0]
        KpS = PA["k_PHOS"].values[0]
        KdS = PA["k_DEPHOS"].values[0]
        S0 = 10**PA["STAT30"].values[0]
        PA["Result"] = ((KbS*C0+KdS/KpS*(KbS*(C0+S0)+(KuS+KpS)))-((KbS*C0+KdS/KpS*(KbS*(C0+S0)+(KuS+KpS)))**2-4*KbS*KdS/KpS*(1+KdS/KpS)*KbS*C0*S0)**(1/2))/(2*KbS*KdS/KpS*(1+KdS/KpS))
    elif PA["STAT_type"].values[0] == "pSTAT1":
        C0 = simulator['IL_RA1_RB1'][-1]+simulator['IL_RA2_RB2'][-1]+simulator['IL_RA1_RA2_RB1'][-1]+simulator['IL_RA1_RA2_RB2'][-1]+simulator['IL_RA1_RB1_RB2'][-1]+simulator['IL_RA2_RB1_RB2'][-1]+2*simulator['IL_RA1_RA2_RB1_RB2'][-1]
        KbS = PA["k_STAT1_f"].values[0]/(Na*PA["vol_cell"].values[0])
        KuS = PA["k_STAT1_b"].values[0]
        KpS = PA["k_PHOS"].values[0]
        KdS = PA["k_DEPHOS"].values[0]
        S0 = 10**PA["STAT10"].values[0]
        PA["Result"] = ((KbS*C0+KdS/KpS*(KbS*(C0+S0)+(KuS+KpS)))-((KbS*C0+KdS/KpS*(KbS*(C0+S0)+(KuS+KpS)))**2-4*KbS*KdS/KpS*(1+KdS/KpS)*KbS*C0*S0)**(1/2))/(2*KbS*KdS/KpS*(1+KdS/KpS))
        
    return PA