from pysb import Model, Parameter, Compartment, Monomer, Parameter, Rule, Initial, Observable
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import cython

def model_function(PA):
    """ Model function to simulate the "Kinetic proofreading" IL-10 signaling model at a specific IL-10 dose, cell type and IL-10 variant using the full ODE system.
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
    
    if PA["STAT_type"].values[0] == "pSTAT3":
        # Initialize model
        Model()

        # Compartment
        Compartment(name='EC', parent=None, dimension=3, size=None)
        Compartment(name='Cell_PM', parent=EC, dimension=2, size=None)
        Compartment(name='Cell_CP', parent=Cell_PM, dimension=3, size=None)

        # Monomers
        Monomer('ILM')
        Monomer('IL', ['bA1', 'bA2', 'bB1', 'bB2','B_t'], {'B_t': ['QB','TB','DB','SB','U']})
        Monomer('RA', ['bA', 'bSTAT3'])
        Monomer('RB', ['bB'])
        Monomer('STAT3', ['b3','P'], {'P': ['u', 'p']})

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

        Parameter('K_STAT3_f', PA["k_STAT3_f"].values[0]/(Na*PA["vol_cell"].values[0]))
        Parameter('K_STAT3_b', PA["k_STAT3_b"].values[0])
        Parameter('K_PHOS', PA["k_PHOS"].values[0])   
        Parameter('K_DEPHOS', PA["k_DEPHOS"].values[0])

        # Complexes and reactions (binding of receptors)
        IL_free = IL(bA1=None, bA2=None, bB1=None, bB2=None, B_t='U')
        ILM_free = ILM()
        Rule('IL_binding', ILM_free + ILM_free | IL_free, *[K_ILM_f, K_ILM_b])

        # Binding of 1st receptor
        IL_RA1_alone = IL(bA1=1, bA2=None, bB1=None, bB2=None, B_t='SB') % RA(bA=1, bSTAT3=None)
        Rule('IL_RA1_alone_binding', IL_free + RA(bA=None, bSTAT3=None)| IL_RA1_alone, *[K_IL_RA_f, K_IL_RA_b])
        IL_RA2_alone = IL(bA1=None, bA2=2, bB1=None, bB2=None, B_t='SB') % RA(bA=2, bSTAT3=None)
        Rule('IL_RA2_alone_binding', IL_free + RA(bA=None, bSTAT3=None) | IL_RA2_alone, *[K_IL_RA_f, K_IL_RA_b])
        IL_RB1_alone = IL(bA1=None, bA2=None, bB1=3, bB2=None, B_t='SB') % RB(bB=3)
        Rule('IL_RB1_binding_alone', IL_free + RB(bB=None) | IL_RB1_alone, *[K_IL_RB_f, K_IL_RB_b])
        IL_RB2_alone = IL(bA1=None, bA2=None, bB1=None, bB2=4, B_t='SB') % RB(bB=4)
        Rule('IL_RB2_binding_alone', IL_free + RB(bB=None) | IL_RB2_alone, *[K_IL_RB_f, K_IL_RB_b])

        # Binding of 2nd receptor
        IL_RA1_RB1_DB = IL(bA1=1, bA2=None, bB1=3, bB2=None, B_t='DB') % RA(bA=1, bSTAT3=None) % RB(bB=3)
        Rule('IL_RA1_wRB1_binding_DB', IL_RB1_alone + RA(bA=None, bSTAT3=None) | IL_RA1_RB1_DB, *[K_IL_RA2_f, K_IL_RA2_b])
        IL_RA1_RB1_S3_DB = IL(bA1=1, bA2=None, bB1=2, bB2=None, B_t='DB') % RA(bA=1, bSTAT3=5) % RB(bB=2) % STAT3(b3=5, P='u')
        Rule('IL_RA1_wRB1_unbinding_DB', IL_RA1_RB1_S3_DB >> IL_RB1_alone + RA(bA=None, bSTAT3=None) + STAT3(b3=None, P='u'), K_IL_RA2_b)
        IL_noRA1_noRB1_SB = IL(bA1=None, bB1=None, B_t='SB')
        IL_RA1_noRB1_DB = IL(bA1=1, bB1=None, B_t='DB') % RA(bA=1, bSTAT3=None) 
        Rule('IL_RA1_noRB1_binding_DB', IL_noRA1_noRB1_SB + RA(bA=None, bSTAT3=None) | IL_RA1_noRB1_DB, *[K_IL_RA2_f, K_IL_RA2_b])

        IL_RA2_RB2_DB = IL(bA1=None, bA2=2, bB1=None, bB2=4, B_t='DB') % RA(bA=2, bSTAT3=None) % RB(bB=4)
        Rule('IL_RA2_wRB2_binding_DB', IL_RB2_alone + RA(bA=None, bSTAT3=None) | IL_RA2_RB2_DB, *[K_IL_RA2_f, K_IL_RA2_b])
        IL_RA2_RB2_S3_DB = IL(bA1=None, bA2=2, bB1=None, bB2=4, B_t='DB') % RA(bA=2, bSTAT3=6) % RB(bB=4) % STAT3(b3=6, P='u')
        Rule('IL_RA2_wRB2_unbinding_DB', IL_RA2_RB2_S3_DB >> IL_RB2_alone + RA(bA=None, bSTAT3=None) + STAT3(b3=None, P='u'), K_IL_RA2_b)
        IL_noRA2_noRB2_SB = IL(bA2=None, bB2=None, B_t='SB')
        IL_RA2_noRB2_DB = IL(bA2=2, bB2=None, B_t='DB') % RA(bA=2, bSTAT3=None) 
        Rule('IL_RA2_noRB2_binding_DB', IL_noRA2_noRB2_SB + RA(bA=None, bSTAT3=None) | IL_RA2_noRB2_DB, *[K_IL_RA2_f, K_IL_RA2_b])

        IL_noRA1_RB1_DB = IL(bA1=None, bB1=3, B_t='DB') % RB(bB=3)
        Rule('IL_noRA1_RB1_binding_DB', IL_noRA1_noRB1_SB + RB(bB=None) | IL_noRA1_RB1_DB, *[K_IL_RB2_f, K_IL_RB2_b])
        Rule('IL_wRA1_RB1_binding_DB', IL_RA1_alone + RB(bB=None) | IL_RA1_RB1_DB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_wRA1_RB1_unbinding_DB', IL_RA1_RB1_S3_DB >> IL_RA1_alone + RB(bB=None) + STAT3(b3=None, P='u'), K_IL_RA_RB_b)

        IL_noRA2_RB2_DB = IL(bA2=None, bB2=4, B_t='DB') % RB(bB=4)
        Rule('IL_noRA2_RB2_binding_DB', IL_noRA2_noRB2_SB + RB(bB=None) | IL_noRA2_RB2_DB, *[K_IL_RB2_f, K_IL_RB2_b])
        Rule('IL_wRA2_RB2_binding_DB', IL_RA2_alone + RB(bB=None) | IL_RA2_RB2_DB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_wRA2_RB2_unbinding_DB', IL_RA2_RB2_S3_DB >> IL_RA2_alone + RB(bB=None) + STAT3(b3=None, P='u'), K_IL_RA_RB_b)


        # Binding of 3rd receptor
        IL_noRA1_RB1_DB = IL(bA1=None, bB1=3, B_t='DB') % RB(bB=3)
        IL_RA1_RB1_TB = IL(bA1=1, bB1=3, B_t='TB') % RA(bA=1, bSTAT3=None) % RB(bB=3)
        Rule('IL_RA1_wRB1_binding_TB', IL_noRA1_RB1_DB + RA(bA=None, bSTAT3=None) | IL_RA1_RB1_TB, *[K_IL_RA2_f, K_IL_RA2_b])
        IL_RA1_RB1_S3_TB = IL(bA1=1, bB1=2, B_t='TB') % RA(bA=1, bSTAT3=5) % RB(bB=2) % STAT3(b3=5, P='u')
        Rule('IL_RA1_wRB1_unbinding_TB', IL_RA1_RB1_S3_TB >> IL_noRA1_RB1_DB + RA(bA=None, bSTAT3=None) + STAT3(b3=None, P='u'), K_IL_RA2_b)
        IL_noRA1_noRB1_DB = IL(bA1=None, bB1=None, B_t='DB')
        IL_RA1_noRB1_TB = IL(bA1=1, bB1=None, B_t='TB') % RA(bA=1, bSTAT3=None) 
        Rule('IL_RA1_noRB1_binding_TB', IL_noRA1_noRB1_DB + RA(bA=None, bSTAT3=None) | IL_RA1_noRB1_TB, *[K_IL_RA2_f, K_IL_RA2_b])

        IL_noRA2_RB2_DB = IL(bA2=None, bB2=4, B_t='DB') % RB(bB=4)
        IL_RA2_RB2_TB = IL(bA2=2, bB2=4, B_t='TB') % RA(bA=2, bSTAT3=None) % RB(bB=4)
        Rule('IL_RA2_wRB2_binding_TB', IL_noRA2_RB2_DB + RA(bA=None, bSTAT3=None) | IL_RA2_RB2_TB, *[K_IL_RA2_f, K_IL_RA2_b])
        IL_RA2_RB2_S3_TB = IL(bA2=2, bB2=4, B_t='TB') % RA(bA=2, bSTAT3=6) % RB(bB=4) % STAT3(b3=6, P='u')
        Rule('IL_RA2_wRB2_unbinding_TB', IL_RA2_RB2_S3_TB >> IL_noRA2_RB2_DB + RA(bA=None, bSTAT3=None) + STAT3(b3=None, P='u'), K_IL_RA2_b)
        IL_noRA2_noRB2_DB = IL(bA2=None, bB2=None, B_t='DB')
        IL_RA2_noRB2_TB = IL(bA2=2, bB2=None, B_t='TB') % RA(bA=2, bSTAT3=None) 
        Rule('IL_RA2_noRB2_binding_TB', IL_noRA2_noRB2_DB + RA(bA=None, bSTAT3=None) | IL_RA2_noRB2_TB, *[K_IL_RA2_f, K_IL_RA2_b])

        IL_noRA1_RB1_TB = IL(bA1=None, bB1=3, B_t='TB') % RB(bB=3)
        Rule('IL_noRA1_RB1_binding_TB', IL_noRA1_noRB1_DB + RB(bB=None) | IL_noRA1_RB1_TB, *[K_IL_RB2_f, K_IL_RB2_b])
        IL_RA1_noRB1_DB = IL(bA1=1, bB1=None, B_t='DB') % RA(bA=1, bSTAT3=None)
        Rule('IL_wRA1_RB1_binding_TB', IL_RA1_noRB1_DB + RB(bB=None) | IL_RA1_RB1_TB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_wRA1_RB1_unbinding_TB', IL_RA1_RB1_S3_TB >> IL_RA1_noRB1_DB + RB(bB=None) + STAT3(b3=None, P='u'), K_IL_RA_RB_b)

        IL_noRA2_RB2_TB = IL(bA2=None, bB2=4, B_t='TB') % RB(bB=4)
        Rule('IL_noRA2_RB2_binding_TB', IL_noRA2_noRB2_DB + RB(bB=None) | IL_noRA2_RB2_TB, *[K_IL_RB2_f, K_IL_RB2_b])
        IL_RA2_noRB2_DB = IL(bA2=2, bB2=None, B_t='DB') % RA(bA=2, bSTAT3=None)
        Rule('IL_wRA2_RB2_binding_TB', IL_RA2_noRB2_DB + RB(bB=None) | IL_RA2_RB2_TB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_wRA2_RB2_unbinding_TB', IL_RA2_RB2_S3_TB >> IL_RA2_noRB2_DB + RB(bB=None) + STAT3(b3=None, P='u'), K_IL_RA_RB_b)


        # Binding of 4th receptor
        IL_RA1_RA2_RB1_RB2 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1, bSTAT3=None) % RA(bA=2, bSTAT3=None) % RB(bB=3) % RB(bB=4)
        IL_RA1_RA2_RB1_RB2_S31 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1, bSTAT3=5) % RA(bA=2, bSTAT3=None) % RB(bB=3) % RB(bB=4) % STAT3(b3=5, P='u')
        IL_RA1_RA2_RB1_RB2_S32 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1, bSTAT3=None) % RA(bA=2, bSTAT3=6) % RB(bB=3) % RB(bB=4) % STAT3(b3=6, P='u')
        IL_RA1_RA2_RB1_RB2_S31_S32 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1, bSTAT3=5) % RA(bA=2, bSTAT3=6) % RB(bB=3) % RB(bB=4) % STAT3(b3=5, P='u') % STAT3(b3=6, P='u')

        IL_noRA1_TB = IL(bA1=None, bA2=2, bB1=3, bB2=4, B_t='TB') % RA(bA=2, bSTAT3=None) % RB(bB=3) % RB(bB=4)
        IL_noRA1_TB_S32 = IL(bA1=None, bA2=2, bB1=3, bB2=4, B_t='TB') % RA(bA=2, bSTAT3=6) % RB(bB=3) % RB(bB=4) % STAT3(b3=6, P='u')
        Rule('IL_RA1_binding_fin', IL_noRA1_TB + RA(bA=None, bSTAT3=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA2_f, K_IL_RA2_b])
        Rule('IL_RA1_binding_fin2', IL_noRA1_TB_S32 + RA(bA=None, bSTAT3=None) | IL_RA1_RA2_RB1_RB2_S32, *[K_IL_RA2_f, K_IL_RA2_b])
        Rule('IL_RA1_unbinding_fin1', IL_RA1_RA2_RB1_RB2_S31 >> IL_noRA1_TB + RA(bA=None, bSTAT3=None) + STAT3(b3=None, P='u'), K_IL_RA2_b)
        Rule('IL_RA1_unbinding_fin2', IL_RA1_RA2_RB1_RB2_S31_S32 >> IL_noRA1_TB_S32 + RA(bA=None, bSTAT3=None) + STAT3(b3=None, P='u'), K_IL_RA2_b)

        IL_noRA2_TB = IL(bA1=1, bA2=None, bB1=3, bB2=4, B_t='TB') % RA(bA=1, bSTAT3=None) % RB(bB=3) % RB(bB=4)
        IL_noRA2_TB_S31 = IL(bA1=1, bA2=None, bB1=3, bB2=4, B_t='TB') % RA(bA=1, bSTAT3=5) % RB(bB=3) % RB(bB=4) % STAT3(b3=5, P='u')
        Rule('IL_RA2_binding_fin', IL_noRA2_TB + RA(bA=None, bSTAT3=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA2_f, K_IL_RA2_b])
        Rule('IL_RA2_binding_fin2', IL_noRA2_TB_S31 + RA(bA=None, bSTAT3=None) | IL_RA1_RA2_RB1_RB2_S31, *[K_IL_RA2_f, K_IL_RA2_b])
        Rule('IL_RA2_unbinding_fin1', IL_RA1_RA2_RB1_RB2_S32 >> IL_noRA2_TB + RA(bA=None, bSTAT3=None) + STAT3(b3=None, P='u'), K_IL_RA2_b)
        Rule('IL_RA2_unbinding_fin2', IL_RA1_RA2_RB1_RB2_S31_S32 >> IL_noRA2_TB_S31 + RA(bA=None, bSTAT3=None) + STAT3(b3=None, P='u'), K_IL_RA2_b)

        IL_noRB1_TB = IL(bA1=1, bA2=2, bB1=None, bB2=4, B_t='TB') % RA(bA=1, bSTAT3=None) % RA(bA=2, bSTAT3=None) % RB(bB=4)
        IL_noRB1_TB_S32 = IL(bA1=1, bA2=2, bB1=None, bB2=4, B_t='TB') % RA(bA=1, bSTAT3=None) % RA(bA=2, bSTAT3=6) % RB(bB=4) % STAT3(b3=6, P='u')
        Rule('IL_RB1_binding_fin', IL_noRB1_TB + RB(bB=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_RB1_binding_fin2', IL_noRB1_TB_S32 + RB(bB=None) | IL_RA1_RA2_RB1_RB2_S32, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_RB1_unbinding_fin1', IL_RA1_RA2_RB1_RB2_S31 >> IL_noRB1_TB + RB(bB=None) + STAT3(b3=None, P='u'), K_IL_RA_RB_b)
        Rule('IL_RB1_unbinding_fin2', IL_RA1_RA2_RB1_RB2_S31_S32 >> IL_noRB1_TB_S32 + RB(bB=None) + STAT3(b3=None, P='u'), K_IL_RA_RB_b)

        IL_noRB2_TB = IL(bA1=1, bA2=2, bB1=3, bB2=None, B_t='TB') % RA(bA=1, bSTAT3=None) % RA(bA=2, bSTAT3=None) % RB(bB=3)
        IL_noRB2_TB_S31 = IL(bA1=1, bA2=2, bB1=3, bB2=None, B_t='TB') % RA(bA=1, bSTAT3=5) % RA(bA=2, bSTAT3=None) % RB(bB=3) % STAT3(b3=5, P='u')
        Rule('IL_RB2_binding_fin', IL_noRB2_TB + RB(bB=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_RB2_binding_fin2', IL_noRB2_TB_S31 + RB(bB=None) | IL_RA1_RA2_RB1_RB2_S31, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_RB2_unbinding_fin1', IL_RA1_RA2_RB1_RB2_S32 >> IL_noRB2_TB + RB(bB=None) + STAT3(b3=None, P='u'), K_IL_RA_RB_b)
        Rule('IL_RB2_unbinding_fin2', IL_RA1_RA2_RB1_RB2_S31_S32 >> IL_noRB2_TB_S31 + RB(bB=None) + STAT3(b3=None, P='u'), K_IL_RA_RB_b)

        # Binding of STAT
        IL_RA1_RB1 = IL(bA1=1, bB1=3) % RA(bA=1, bSTAT3=None) % RB(bB=3)
        IL_RA1_RB1_S31 = IL(bA1=1, bB1=3) % RA(bA=1, bSTAT3=5) % RB(bB=3) % STAT3(b3=5, P='u')
        Rule('IL_RA1_RB1_S3_binding', IL_RA1_RB1 + STAT3(b3=None, P='u') | IL_RA1_RB1_S31, *[K_STAT3_f, K_STAT3_b])
        IL_RA2_RB2 = IL(bA2=2, bB2=4) % RA(bA=2, bSTAT3=None) % RB(bB=4)
        IL_RA2_RB2_S32 = IL(bA2=2, bB2=4) % RA(bA=2, bSTAT3=6) % RB(bB=4) % STAT3(b3=6, P='u')
        Rule('IL_RA2_RB2_S3_binding', IL_RA2_RB2 + STAT3(b3=None, P='u') | IL_RA2_RB2_S32, *[K_STAT3_f, K_STAT3_b])

        Rule('IL_RA1_RB1_S3_phosphorylation', IL_RA1_RB1_S31 >> IL_RA1_RB1 + STAT3(b3=None, P='p'), K_PHOS)
        Rule('IL_RA2_RB2_S3_phosphorylation', IL_RA2_RB2_S32 >> IL_RA2_RB2 + STAT3(b3=None, P='p'), K_PHOS)
        Rule('STAT3_dephosphorilation', STAT3(b3=None, P='p') >> STAT3(b3=None, P='u'), K_DEPHOS)

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
        Initial(RA(bA=None, bSTAT3=None) ** Cell_PM, RA_0)
        Initial(RB(bB=None) ** Cell_PM, RB_0)

        Parameter('STAT_3_0', 10**PA["STAT30"].values[0])
        Initial(STAT3(b3=None, P='u') ** Cell_CP, STAT_3_0)

        # Simulation
        Observable('ILM_free', ILM() ** EC)
        Observable('IL_free', IL(bA1=None, bA2=None, bB1=None, bB2=None, B_t='U') ** EC)
        Observable('RA_free', RA(bA=None, bSTAT3=None) ** Cell_PM)
        Observable('RB_free', RB(bB=None) ** Cell_PM)
        Observable('S3_free', STAT3(b3=None, P='u') ** Cell_CP)

        Observable('IL_RA1', IL(bA1=1, bA2=None, bB1=None, bB2=None, B_t='SB') ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM)
        Observable('IL_RA2', IL(bA1=None, bA2=1, bB1=None, bB2=None, B_t='SB') ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM)
        Observable('IL_RB1', IL(bA1=None, bA2=None, bB1=1, bB2=None, B_t='SB') ** EC % RB(bB=1) ** Cell_PM)
        Observable('IL_RB2', IL(bA1=None, bA2=None, bB1=None, bB2=1, B_t='SB') ** EC % RB(bB=1) ** Cell_PM)

        Observable('IL_RA1_RA2', IL(bA1=1, bA2=2, bB1=None, bB2=None, B_t='DB') ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RA(bA=2, bSTAT3=None) ** Cell_PM)
        Observable('IL_RA1_RB1', IL(bA1=1, bA2=None, bB1=2, bB2=None, B_t='DB') ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RB(bB=2) ** Cell_PM)
        Observable('IL_RA1_RB2', IL(bA1=1, bA2=None, bB1=None, bB2=2, B_t='DB') ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RB(bB=2) ** Cell_PM)
        Observable('IL_RA2_RB1', IL(bA1=None, bA2=1, bB1=2, bB2=None, B_t='DB') ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RB(bB=2) ** Cell_PM)
        Observable('IL_RA2_RB2', IL(bA1=None, bA2=1, bB1=None, bB2=2, B_t='DB') ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RB(bB=2) ** Cell_PM)
        Observable('IL_RB1_RB2', IL(bA1=None, bA2=None, bB1=1, bB2=2, B_t='DB') ** EC % RB(bB=1) ** Cell_PM % RB(bB=2) ** Cell_PM)

        Observable('IL_RA1_RA2_RB1', IL(bA1=1, bA2=2, bB1=3, bB2=None, B_t='TB') ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RA(bA=2, bSTAT3=None) ** Cell_PM % RB(bB=3) ** Cell_PM)
        Observable('IL_RA1_RA2_RB2', IL(bA1=1, bA2=2, bB1=None, bB2=3, B_t='TB') ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RA(bA=2, bSTAT3=None) ** Cell_PM % RB(bB=3) ** Cell_PM)
        Observable('IL_RA1_RB1_RB2', IL(bA1=1, bA2=None, bB1=2, bB2=3, B_t='TB') ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RB(bB=2) ** Cell_PM % RB(bB=3) ** Cell_PM)
        Observable('IL_RA2_RB1_RB2', IL(bA1=None, bA2=1, bB1=2, bB2=3, B_t='TB') ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RB(bB=2) ** Cell_PM % RB(bB=3) ** Cell_PM)

        Observable('IL_RA1_RA2_RB1_RB2', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RA(bA=2, bSTAT3=None) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM)

        Observable('IL_RA1_RB1_S31', IL(bA1=1, bA2=None, bB1=2, bB2=None, B_t='DB') ** EC % RA(bA=1, bSTAT3=5) ** Cell_PM % RB(bB=2) ** Cell_PM % STAT3(b3=5, P='u') ** Cell_CP)
        Observable('IL_RA2_RB2_S32', IL(bA1=None, bA2=1, bB1=None, bB2=2, B_t='DB') ** EC % RA(bA=1, bSTAT3=6) ** Cell_PM % RB(bB=2) ** Cell_PM % STAT3(b3=6, P='u') ** Cell_CP)

        Observable('IL_RA1_RA2_RB1_S31', IL(bA1=1, bA2=2, bB1=3, bB2=None, B_t='TB') ** EC % RA(bA=1, bSTAT3=5) ** Cell_PM % RA(bA=2, bSTAT3=None) ** Cell_PM % RB(bB=3) ** Cell_PM % STAT3(b3=5, P='u') ** Cell_CP)
        Observable('IL_RA1_RA2_RB2_S32', IL(bA1=1, bA2=2, bB1=None, bB2=3, B_t='TB') ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RA(bA=2, bSTAT3=6) ** Cell_PM % RB(bB=3) ** Cell_PM % STAT3(b3=6, P='u') ** Cell_CP)
        Observable('IL_RA1_RB1_RB2_S31', IL(bA1=1, bA2=None, bB1=2, bB2=3, B_t='TB') ** EC % RA(bA=1, bSTAT3=5) ** Cell_PM % RB(bB=2) ** Cell_PM % RB(bB=3) ** Cell_PM % STAT3(b3=5, P='u') ** Cell_CP)
        Observable('IL_RA2_RB1_RB2_S32', IL(bA1=None, bA2=1, bB1=2, bB2=3, B_t='TB') ** EC % RA(bA=1, bSTAT3=6) ** Cell_PM % RB(bB=2) ** Cell_PM % RB(bB=3) ** Cell_PM % STAT3(b3=6, P='u') ** Cell_CP)

        Observable('IL_RA1_RA2_RB1_RB2_S31', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bSTAT3=5) ** Cell_PM % RA(bA=2, bSTAT3=None) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % STAT3(b3=5, P='u') ** Cell_CP)
        Observable('IL_RA1_RA2_RB1_RB2_S32', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RA(bA=2, bSTAT3=6) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % STAT3(b3=6, P='u') ** Cell_CP)
        Observable('IL_RA1_RA2_RB1_RB2_S31_S32', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bSTAT3=5) ** Cell_PM % RA(bA=2, bSTAT3=6) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % STAT3(b3=5, P='u') ** Cell_CP % STAT3(b3=6, P='u') ** Cell_CP)

        Observable('STAT3P', STAT3(b3=None, P='p') ** Cell_CP)
    
        dT, Tf = func_time(PA["IL0"].values[0])
        t = np.arange(0,Tf,dT/500)
        simulator = ScipyOdeSimulator(model, tspan=t, compiler='cython', integrator = 'lsoda').run().all
        PA["Result"] = simulator['STAT3P'][-1]

    elif PA["STAT_type"].values[0] == "pSTAT1":
        # Initialize model
        Model()

        # Compartment
        Compartment(name='EC', parent=None, dimension=3, size=None)
        Compartment(name='Cell_PM', parent=EC, dimension=2, size=None)
        Compartment(name='Cell_CP', parent=Cell_PM, dimension=3, size=None)

        # Monomers
        Monomer('ILM')
        Monomer('IL', ['bA1', 'bA2', 'bB1', 'bB2','B_t'], {'B_t': ['QB','TB','DB','SB','U']})
        Monomer('RA', ['bA', 'bM'])
        Monomer('RB', ['bB'])
        Monomer('M', ['bRA','bSTAT1'])
        Monomer('STAT1', ['b1','P'], {'P': ['u', 'p']})

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

        Parameter('K_M_f', PA["k_M_f"].values[0]/(Na*PA["vol_cell"].values[0]))
        Parameter('K_M_b', PA["k_M_b"].values[0])
        Parameter('K_STAT1_f', PA["k_STAT1_f"].values[0]/(Na*PA["vol_cell"].values[0]))
        Parameter('K_STAT1_b', PA["k_STAT1_b"].values[0])
        Parameter('K_PHOS', PA["k_PHOS"].values[0])   
        Parameter('K_DEPHOS', PA["k_DEPHOS"].values[0])

        # Complexes and reactions (binding of receptors)
        IL_free = IL(bA1=None, bA2=None, bB1=None, bB2=None, B_t='U')
        ILM_free = ILM()
        Rule('IL_binding', ILM_free + ILM_free | IL_free, *[K_ILM_f, K_ILM_b])

        # Binding of 1st receptor
        IL_RA1_alone = IL(bA1=1, bA2=None, bB1=None, bB2=None, B_t='SB') % RA(bA=1, bM=None)
        Rule('IL_RA1_alone_binding', IL_free + RA(bA=None, bM=None)| IL_RA1_alone, *[K_IL_RA_f, K_IL_RA_b])
        IL_RA2_alone = IL(bA1=None, bA2=2, bB1=None, bB2=None, B_t='SB') % RA(bA=2, bM=None)
        Rule('IL_RA2_alone_binding', IL_free + RA(bA=None, bM=None) | IL_RA2_alone, *[K_IL_RA_f, K_IL_RA_b])
        IL_RB1_alone = IL(bA1=None, bA2=None, bB1=3, bB2=None, B_t='SB') % RB(bB=3)
        Rule('IL_RB1_binding_alone', IL_free + RB(bB=None) | IL_RB1_alone, *[K_IL_RB_f, K_IL_RB_b])
        IL_RB2_alone = IL(bA1=None, bA2=None, bB1=None, bB2=4, B_t='SB') % RB(bB=4)
        Rule('IL_RB2_binding_alone', IL_free + RB(bB=None) | IL_RB2_alone, *[K_IL_RB_f, K_IL_RB_b])

        # Binding of 2nd receptor
        IL_RA1_RB1_DB = IL(bA1=1, bA2=None, bB1=3, bB2=None, B_t='DB') % RA(bA=1, bM=None) % RB(bB=3)
        Rule('IL_RA1_wRB1_binding_DB', IL_RB1_alone + RA(bA=None, bM=None) | IL_RA1_RB1_DB, *[K_IL_RA2_f, K_IL_RA2_b])
        IL_RA1_RB1_M1_DB = IL(bA1=1, bA2=None, bB1=2, bB2=None, B_t='DB') % RA(bA=1, bM=5) % RB(bB=2) % M(bRA=5, bSTAT1=None)
        IL_RA1_RB1_MS1_DB = IL(bA1=1, bA2=None, bB1=2, bB2=None, B_t='DB') % RA(bA=1, bM=5) % RB(bB=2) % M(bRA=5, bSTAT1=7) % STAT1(b1=7, P='u')
        Rule('IL_RA1_wRB1_unbinding_DB', IL_RA1_RB1_M1_DB >> IL_RB1_alone + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None), K_IL_RA2_b) 
        Rule('IL_RA1_wRB1_unbinding_DB2', IL_RA1_RB1_MS1_DB >> IL_RB1_alone + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA2_b)
        IL_noRA1_noRB1_SB = IL(bA1=None, bB1=None, B_t='SB')
        IL_RA1_noRB1_DB = IL(bA1=1, bB1=None, B_t='DB') % RA(bA=1, bM=None) 
        Rule('IL_RA1_noRB1_binding_DB', IL_noRA1_noRB1_SB + RA(bA=None, bM=None) | IL_RA1_noRB1_DB, *[K_IL_RA2_f, K_IL_RA2_b])

        IL_RA2_RB2_DB = IL(bA1=None, bA2=2, bB1=None, bB2=4, B_t='DB') % RA(bA=2, bM=None) % RB(bB=4)
        Rule('IL_RA2_wRB2_binding_DB', IL_RB2_alone + RA(bA=None, bM=None) | IL_RA2_RB2_DB, *[K_IL_RA2_f, K_IL_RA2_b])
        IL_RA2_RB2_M2_DB = IL(bA1=None, bA2=2, bB1=None, bB2=4, B_t='DB') % RA(bA=2, bM=6) % RB(bB=4) % M(bRA=6, bSTAT1=None)
        IL_RA2_RB2_MS2_DB = IL(bA1=None, bA2=2, bB1=None, bB2=4, B_t='DB') % RA(bA=2, bM=6) % RB(bB=4) % M(bRA=6, bSTAT1=8) % STAT1(b1=8, P='u')
        Rule('IL_RA2_wRB2_unbinding_DB', IL_RA2_RB2_M2_DB >> IL_RB2_alone + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None), K_IL_RA2_b)
        Rule('IL_RA2_wRB2_unbinding_DB2', IL_RA2_RB2_MS2_DB >> IL_RB2_alone + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA2_b)
        IL_noRA2_noRB2_SB = IL(bA2=None, bB2=None, B_t='SB')
        IL_RA2_noRB2_DB = IL(bA2=2, bB2=None, B_t='DB') % RA(bA=2, bM=None) 
        Rule('IL_RA2_noRB2_binding_DB', IL_noRA2_noRB2_SB + RA(bA=None, bM=None) | IL_RA2_noRB2_DB, *[K_IL_RA2_f, K_IL_RA2_b])

        IL_noRA1_RB1_DB = IL(bA1=None, bB1=3, B_t='DB') % RB(bB=3)
        Rule('IL_noRA1_RB1_binding_DB', IL_noRA1_noRB1_SB + RB(bB=None) | IL_noRA1_RB1_DB, *[K_IL_RB2_f, K_IL_RB2_b])
        Rule('IL_wRA1_RB1_binding_DB', IL_RA1_alone + RB(bB=None) | IL_RA1_RB1_DB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_wRA1_RB1_unbinding_DB', IL_RA1_RB1_M1_DB >> IL_RA1_alone + RB(bB=None) + M(bRA=None, bSTAT1=None), K_IL_RA_RB_b)
        Rule('IL_wRA1_RB1_unbinding_DB2', IL_RA1_RB1_MS1_DB >> IL_RA1_alone + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_b)

        IL_noRA2_RB2_DB = IL(bA2=None, bB2=4, B_t='DB') % RB(bB=4)
        Rule('IL_noRA2_RB2_binding_DB', IL_noRA2_noRB2_SB + RB(bB=None) | IL_noRA2_RB2_DB, *[K_IL_RB2_f, K_IL_RB2_b])
        Rule('IL_wRA2_RB2_binding_DB', IL_RA2_alone + RB(bB=None) | IL_RA2_RB2_DB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_wRA2_RB2_unbinding_DB', IL_RA2_RB2_M2_DB >> IL_RA2_alone + RB(bB=None) + M(bRA=None, bSTAT1=None), K_IL_RA_RB_b)
        Rule('IL_wRA2_RB2_unbinding_DB2', IL_RA2_RB2_MS2_DB >> IL_RA2_alone + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_b)

        # Binding of 3rd receptor
        IL_noRA1_RB1_DB = IL(bA1=None, bB1=3, B_t='DB') % RB(bB=3)
        IL_RA1_RB1_TB = IL(bA1=1, bB1=3, B_t='TB') % RA(bA=1, bM=None) % RB(bB=3)
        Rule('IL_RA1_wRB1_binding_TB', IL_noRA1_RB1_DB + RA(bA=None, bM=None) | IL_RA1_RB1_TB, *[K_IL_RA2_f, K_IL_RA2_b])
        IL_RA1_RB1_M1_TB = IL(bA1=1, bB1=2, B_t='TB') % RA(bA=1, bM=5) % RB(bB=2) % M(bRA=5, bSTAT1=None)
        IL_RA1_RB1_MS1_TB = IL(bA1=1, bB1=2, B_t='TB') % RA(bA=1, bM=5) % RB(bB=2) % M(bRA=5, bSTAT1=7) % STAT1(b1=7, P='u')
        Rule('IL_RA1_wRB1_unbinding_TB', IL_RA1_RB1_M1_TB >> IL_noRA1_RB1_DB + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None), K_IL_RA2_b)
        Rule('IL_RA1_wRB1_unbinding_TB2', IL_RA1_RB1_MS1_TB >> IL_noRA1_RB1_DB + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA2_b)
        IL_noRA1_noRB1_DB = IL(bA1=None, bB1=None, B_t='DB')
        IL_RA1_noRB1_TB = IL(bA1=1, bB1=None, B_t='TB') % RA(bA=1, bM=None) 
        Rule('IL_RA1_noRB1_binding_TB', IL_noRA1_noRB1_DB + RA(bA=None, bM=None) | IL_RA1_noRB1_TB, *[K_IL_RA2_f, K_IL_RA2_b])

        IL_noRA2_RB2_DB = IL(bA2=None, bB2=4, B_t='DB') % RB(bB=4)
        IL_RA2_RB2_TB = IL(bA2=2, bB2=4, B_t='TB') % RA(bA=2, bM=None) % RB(bB=4)
        Rule('IL_RA2_wRB2_binding_TB', IL_noRA2_RB2_DB + RA(bA=None, bM=None) | IL_RA2_RB2_TB, *[K_IL_RA2_f, K_IL_RA2_b])
        IL_RA2_RB2_M2_TB = IL(bA2=2, bB2=4, B_t='TB') % RA(bA=2, bM=6) % RB(bB=4) % M(bRA=6, bSTAT1=None)
        IL_RA2_RB2_MS2_TB = IL(bA2=2, bB2=4, B_t='TB') % RA(bA=2, bM=6) % RB(bB=4) % M(bRA=6, bSTAT1=8) % STAT1(b1=8, P='u')
        Rule('IL_RA2_wRB2_unbinding_TB', IL_RA2_RB2_M2_TB >> IL_noRA2_RB2_DB + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None), K_IL_RA2_b)
        Rule('IL_RA2_wRB2_unbinding_TB2', IL_RA2_RB2_MS2_TB >> IL_noRA2_RB2_DB + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA2_b)
        IL_noRA2_noRB2_DB = IL(bA2=None, bB2=None, B_t='DB')
        IL_RA2_noRB2_TB = IL(bA2=2, bB2=None, B_t='TB') % RA(bA=2, bM=None) 
        Rule('IL_RA2_noRB2_binding_TB', IL_noRA2_noRB2_DB + RA(bA=None, bM=None) | IL_RA2_noRB2_TB, *[K_IL_RA2_f, K_IL_RA2_b])

        IL_noRA1_RB1_TB = IL(bA1=None, bB1=3, B_t='TB') % RB(bB=3)
        Rule('IL_noRA1_RB1_binding_TB', IL_noRA1_noRB1_DB + RB(bB=None) | IL_noRA1_RB1_TB, *[K_IL_RB2_f, K_IL_RB2_b])
        IL_RA1_noRB1_DB = IL(bA1=1, bB1=None, B_t='DB') % RA(bA=1, bM=None)
        Rule('IL_wRA1_RB1_binding_TB', IL_RA1_noRB1_DB + RB(bB=None) | IL_RA1_RB1_TB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_wRA1_RB1_unbinding_TB', IL_RA1_RB1_M1_TB >> IL_RA1_noRB1_DB + RB(bB=None) + M(bRA=None, bSTAT1=None), K_IL_RA_RB_b)
        Rule('IL_wRA1_RB1_unbinding_TB2', IL_RA1_RB1_MS1_TB >> IL_RA1_noRB1_DB + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_b)

        IL_noRA2_RB2_TB = IL(bA2=None, bB2=4, B_t='TB') % RB(bB=4)
        Rule('IL_noRA2_RB2_binding_TB', IL_noRA2_noRB2_DB + RB(bB=None) | IL_noRA2_RB2_TB, *[K_IL_RB2_f, K_IL_RB2_b])
        IL_RA2_noRB2_DB = IL(bA2=2, bB2=None, B_t='DB') % RA(bA=2, bM=None)
        Rule('IL_wRA2_RB2_binding_TB', IL_RA2_noRB2_DB + RB(bB=None) | IL_RA2_RB2_TB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_wRA2_RB2_unbinding_TB', IL_RA2_RB2_M2_TB >> IL_RA2_noRB2_DB + RB(bB=None) + M(bRA=None, bSTAT1=None), K_IL_RA_RB_b)
        Rule('IL_wRA2_RB2_unbinding_TB2', IL_RA2_RB2_MS2_TB >> IL_RA2_noRB2_DB + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_b)

        # Binding of 4th receptor
        IL_RA1_RA2_RB1_RB2 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1, bM=None) % RA(bA=2, bM=None) % RB(bB=3) % RB(bB=4)
        IL_RA1_RA2_RB1_RB2_M1 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1, bM=5) % RA(bA=2, bM=None) % RB(bB=3) % RB(bB=4) % M(bRA=5, bSTAT1=None)
        IL_RA1_RA2_RB1_RB2_M2 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1, bM=None) % RA(bA=2, bM=6) % RB(bB=3) % RB(bB=4) % M(bRA=6, bSTAT1=None)
        IL_RA1_RA2_RB1_RB2_MS1 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1, bM=5) % RA(bA=2, bM=None) % RB(bB=3) % RB(bB=4) % M(bRA=5, bSTAT1=7) % STAT1(b1=7, P='u')
        IL_RA1_RA2_RB1_RB2_MS2 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1, bM=None) % RA(bA=2, bM=6) % RB(bB=3) % RB(bB=4) % M(bRA=6, bSTAT1=8) % STAT1(b1=8, P='u')
        IL_RA1_RA2_RB1_RB2_M1_M2 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1, bM=5) % RA(bA=2, bM=6) % RB(bB=3) % RB(bB=4) % M(bRA=5, bSTAT1=None) % M(bRA=6, bSTAT1=None)
        IL_RA1_RA2_RB1_RB2_MS1_M2 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1, bM=5) % RA(bA=2, bM=6) % RB(bB=3) % RB(bB=4) % M(bRA=5, bSTAT1=7) % M(bRA=6, bSTAT1=None) % STAT1(b1=7, P='u')
        IL_RA1_RA2_RB1_RB2_M1_MS2 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1, bM=5) % RA(bA=2, bM=6) % RB(bB=3) % RB(bB=4) % M(bRA=5, bSTAT1=None) % M(bRA=6, bSTAT1=8) % STAT1(b1=8, P='u')
        IL_RA1_RA2_RB1_RB2_MS1_MS2 = IL(bA1=1, bA2=2, bB1=3, bB2=4, B_t='QB') % RA(bA=1, bM=5) % RA(bA=2, bM=6) % RB(bB=3) % RB(bB=4) % M(bRA=5, bSTAT1=7) % M(bRA=6, bSTAT1=8) % STAT1(b1=7, P='u') % STAT1(b1=8, P='u')


        IL_noRA1_TB = IL(bA1=None, bA2=2, bB1=3, bB2=4, B_t='TB') % RA(bA=2, bM=None) % RB(bB=3) % RB(bB=4)
        IL_noRA1_TB_M2 = IL(bA1=None, bA2=2, bB1=3, bB2=4, B_t='TB') % RA(bA=2, bM=6) % RB(bB=3) % RB(bB=4) % M(bRA=6, bSTAT1=None)
        IL_noRA1_TB_MS2 = IL(bA1=None, bA2=2, bB1=3, bB2=4, B_t='TB') % RA(bA=2, bM=6) % RB(bB=3) % RB(bB=4) % M(bRA=6, bSTAT1=8) % STAT1(b1=8, P='u')
        Rule('IL_RA1_binding_fin', IL_noRA1_TB + RA(bA=None, bM=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA2_f, K_IL_RA2_b])
        Rule('IL_RA1_unbinding_fin1', IL_RA1_RA2_RB1_RB2_M1 >> IL_noRA1_TB + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None), K_IL_RA2_b)
        Rule('IL_RA1_binding_fin2', IL_noRA1_TB_M2 + RA(bA=None, bM=None) | IL_RA1_RA2_RB1_RB2_M2, *[K_IL_RA2_f, K_IL_RA2_b])
        Rule('IL_RA1_unbinding_fin2', IL_RA1_RA2_RB1_RB2_MS1 >> IL_noRA1_TB + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA2_b)
        Rule('IL_RA1_binding_fin3', IL_noRA1_TB_MS2 + RA(bA=None, bM=None) | IL_RA1_RA2_RB1_RB2_MS2, *[K_IL_RA2_f, K_IL_RA2_b])
        Rule('IL_RA1_unbinding_fin3', IL_RA1_RA2_RB1_RB2_M1_M2 >> IL_noRA1_TB_M2 + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None), K_IL_RA2_b)
        Rule('IL_RA1_unbinding_fin4', IL_RA1_RA2_RB1_RB2_MS1_M2 >> IL_noRA1_TB_M2 + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA2_b)
        Rule('IL_RA1_unbinding_fin5', IL_RA1_RA2_RB1_RB2_M1_MS2 >> IL_noRA1_TB_MS2 + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None), K_IL_RA2_b)
        Rule('IL_RA1_unbinding_fin6', IL_RA1_RA2_RB1_RB2_MS1_MS2 >> IL_noRA1_TB_MS2 + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA2_b)

        IL_noRA2_TB = IL(bA1=1, bA2=None, bB1=3, bB2=4, B_t='TB') % RA(bA=1, bM=None) % RB(bB=3) % RB(bB=4)
        IL_noRA2_TB_M1 = IL(bA1=1, bA2=None, bB1=3, bB2=4, B_t='TB') % RA(bA=1, bM=5) % RB(bB=3) % RB(bB=4) % M(bRA=5, bSTAT1=None)
        IL_noRA2_TB_MS1 = IL(bA1=1, bA2=None, bB1=3, bB2=4, B_t='TB') % RA(bA=1, bM=5) % RB(bB=3) % RB(bB=4) % M(bRA=5, bSTAT1=7) % STAT1(b1=7, P='u')
        Rule('IL_RA2_binding_fin', IL_noRA2_TB + RA(bA=None, bM=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA2_f, K_IL_RA2_b])
        Rule('IL_RA2_binding_fin2', IL_noRA2_TB_M1 + RA(bA=None, bM=None) | IL_RA1_RA2_RB1_RB2_M1, *[K_IL_RA2_f, K_IL_RA2_b])
        Rule('IL_RA2_unbinding_fin1', IL_RA1_RA2_RB1_RB2_M2 >> IL_noRA2_TB + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None), K_IL_RA2_b)
        Rule('IL_RA2_binding_fin3', IL_noRA2_TB_MS1 + RA(bA=None, bM=None) | IL_RA1_RA2_RB1_RB2_MS1, *[K_IL_RA2_f, K_IL_RA2_b])
        Rule('IL_RA2_unbinding_fin2', IL_RA1_RA2_RB1_RB2_MS2 >> IL_noRA2_TB + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA2_b)
        Rule('IL_RA2_unbinding_fin3', IL_RA1_RA2_RB1_RB2_M1_M2 >> IL_noRA2_TB_M1 + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None), K_IL_RA2_b)
        Rule('IL_RA2_unbinding_fin4', IL_RA1_RA2_RB1_RB2_MS1_M2 >> IL_noRA2_TB_MS1 + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None), K_IL_RA2_b)
        Rule('IL_RA2_unbinding_fin5', IL_RA1_RA2_RB1_RB2_M1_MS2 >> IL_noRA2_TB_M1 + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA2_b)
        Rule('IL_RA2_unbinding_fin6', IL_RA1_RA2_RB1_RB2_MS1_MS2 >> IL_noRA2_TB_MS1 + RA(bA=None, bM=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA2_b)

        IL_noRB1_TB = IL(bA1=1, bA2=2, bB1=None, bB2=4, B_t='TB') % RA(bA=1, bM=None) % RA(bA=2, bM=None) % RB(bB=4)
        IL_noRB1_TB_M2 = IL(bA1=1, bA2=2, bB1=None, bB2=4, B_t='TB') % RA(bA=1, bM=None) % RA(bA=2, bM=6) % RB(bB=4) % M(bRA=6, bSTAT1=None)
        IL_noRB1_TB_MS2 = IL(bA1=1, bA2=2, bB1=None, bB2=4, B_t='TB') % RA(bA=1, bM=None) % RA(bA=2, bM=6) % RB(bB=4) % M(bRA=6, bSTAT1=8) % STAT1(b1=8, P='u')
        Rule('IL_RB1_binding_fin', IL_noRB1_TB + RB(bB=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_RB1_unbinding_fin1', IL_RA1_RA2_RB1_RB2_M1 >> IL_noRB1_TB + RB(bB=None) + M(bRA=None, bSTAT1=None), K_IL_RA_RB_b)
        Rule('IL_RB1_binding_fin2', IL_noRB1_TB_M2 + RB(bB=None) | IL_RA1_RA2_RB1_RB2_M2, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_RB1_unbinding_fin2', IL_RA1_RA2_RB1_RB2_MS1 >> IL_noRB1_TB + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_b)
        Rule('IL_RB1_binding_fin3', IL_noRB1_TB_MS2 + RB(bB=None) | IL_RA1_RA2_RB1_RB2_MS2, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_RB1_unbinding_fin3', IL_RA1_RA2_RB1_RB2_M1_M2 >> IL_noRB1_TB_M2 + RB(bB=None) + M(bRA=None, bSTAT1=None), K_IL_RA_RB_b)
        Rule('IL_RB1_unbinding_fin4', IL_RA1_RA2_RB1_RB2_MS1_M2 >> IL_noRB1_TB_M2 + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_b)
        Rule('IL_RB1_unbinding_fin5', IL_RA1_RA2_RB1_RB2_M1_MS2 >> IL_noRB1_TB_MS2 + RB(bB=None) + M(bRA=None, bSTAT1=None), K_IL_RA_RB_b)
        Rule('IL_RB1_unbinding_fin6', IL_RA1_RA2_RB1_RB2_MS1_MS2 >> IL_noRB1_TB_MS2 + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_b)

        IL_noRB2_TB = IL(bA1=1, bA2=2, bB1=3, bB2=None, B_t='TB') % RA(bA=1, bM=None) % RA(bA=2, bM=None) % RB(bB=3)
        IL_noRB2_TB_M1 = IL(bA1=1, bA2=2, bB1=3, bB2=None, B_t='TB') % RA(bA=1, bM=5) % RA(bA=2, bM=None) % RB(bB=3) % M(bRA=5, bSTAT1=None)
        IL_noRB2_TB_MS1 = IL(bA1=1, bA2=2, bB1=3, bB2=None, B_t='TB') % RA(bA=1, bM=5) % RA(bA=2, bM=None) % RB(bB=3) % M(bRA=5, bSTAT1=7) % STAT1(b1=7, P='u')
        Rule('IL_RB2_binding_fin', IL_noRB2_TB + RB(bB=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_RB2_binding_fin2', IL_noRB2_TB_M1 + RB(bB=None) | IL_RA1_RA2_RB1_RB2_M1, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_RB2_unbinding_fin1', IL_RA1_RA2_RB1_RB2_M2 >> IL_noRB2_TB + RB(bB=None) + M(bRA=None, bSTAT1=None), K_IL_RA_RB_b)
        Rule('IL_RB2_binding_fin3', IL_noRB2_TB_MS1 + RB(bB=None) | IL_RA1_RA2_RB1_RB2_MS1, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
        Rule('IL_RB2_unbinding_fin2', IL_RA1_RA2_RB1_RB2_MS2 >> IL_noRB2_TB + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_b)
        Rule('IL_RB2_unbinding_fin3', IL_RA1_RA2_RB1_RB2_M1_M2 >> IL_noRB2_TB_M1 + RB(bB=None) + M(bRA=None, bSTAT1=None), K_IL_RA_RB_b)
        Rule('IL_RB2_unbinding_fin4', IL_RA1_RA2_RB1_RB2_MS1_M2 >> IL_noRB2_TB_MS1 + RB(bB=None) + M(bRA=None, bSTAT1=None), K_IL_RA_RB_b)
        Rule('IL_RB2_unbinding_fin5', IL_RA1_RA2_RB1_RB2_M1_MS2 >> IL_noRB2_TB_M1 + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_b)
        Rule('IL_RB2_unbinding_fin6', IL_RA1_RA2_RB1_RB2_MS1_MS2 >> IL_noRB2_TB_MS1 + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_b)

        # Binding of STAT
        IL_RA1_RB1 = IL(bA1=1, bB1=3) % RA(bA=1, bM=None) % RB(bB=3)
        IL_RA1_RB1_M1 = IL(bA1=1, bB1=3) % RA(bA=1, bM=5) % RB(bB=3) % M(bRA=5, bSTAT1=None)
        IL_RA1_RB1_MS1 = IL(bA1=1, bB1=3) % RA(bA=1, bM=5) % RB(bB=3) % M(bRA=5, bSTAT1=7) % STAT1(b1=7, P='u')
        Rule('IL_RA1_RB1_M1_binding', IL_RA1_RB1 + M(bRA=None, bSTAT1=None) | IL_RA1_RB1_M1, *[K_M_f, K_M_b])
        Rule('IL_RA1_RB1_MS1_binding', IL_RA1_RB1_M1 + STAT1(b1=None, P='u') | IL_RA1_RB1_MS1, *[K_STAT1_f, K_STAT1_b])
        Rule('IL_RA1_RB1_MS1_unbinding', IL_RA1_RB1_MS1 >> IL_RA1_RB1 + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_M_b)

        IL_RA2_RB2 = IL(bA2=2, bB2=4) % RA(bA=2, bM=None) % RB(bB=4)
        IL_RA2_RB2_M2 = IL(bA2=2, bB2=4) % RA(bA=2, bM=6) % RB(bB=4) % M(bRA=6, bSTAT1=None)
        IL_RA2_RB2_MS2 = IL(bA2=2, bB2=4) % RA(bA=2, bM=6) % RB(bB=4) % M(bRA=6, bSTAT1=8) % STAT1(b1=8, P='u')
        Rule('IL_RA2_RB2_M2_binding', IL_RA2_RB2 + M(bRA=None, bSTAT1=None) | IL_RA2_RB2_M2, *[K_M_f, K_M_b])
        Rule('IL_RA2_RB2_MS2_binding', IL_RA2_RB2_M2 + STAT1(b1=None, P='u') | IL_RA2_RB2_MS2, *[K_STAT1_f, K_STAT1_b])
        Rule('IL_RA2_RB2_MS2_unbinding', IL_RA2_RB2_MS2 >> IL_RA2_RB2 + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_M_b)

        Rule('IL_RA1_RB1_S1_phosphorylation', IL_RA1_RB1_MS1 >> IL_RA1_RB1_M1 + STAT1(b1=None, P='p'), K_PHOS)
        Rule('IL_RA2_RB2_S2_phosphorylation', IL_RA2_RB2_MS2 >> IL_RA2_RB2_M2 + STAT1(b1=None, P='p'), K_PHOS)
        Rule('STAT1_dephosphorilation', STAT1(b1=None, P='p') >> STAT1(b1=None, P='u'), K_DEPHOS)

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
        Initial(RA(bA=None, bM=None) ** Cell_PM, RA_0)
        Initial(RB(bB=None) ** Cell_PM, RB_0)

        Parameter('STAT_1_0', 10**PA["STAT10"].values[0])
        Initial(STAT1(b1=None, P='u') ** Cell_CP, STAT_1_0)
        Parameter('M_0', PA["M0"].values[0])
        Initial(M(bRA=None,bSTAT1=None) ** Cell_CP, M_0)

        # Simulation
        Observable('ILM_free', ILM() ** EC)
        Observable('IL_free', IL(bA1=None, bA2=None, bB1=None, bB2=None, B_t='U') ** EC)
        Observable('RA_free', RA(bA=None, bM=None) ** Cell_PM)
        Observable('RB_free', RB(bB=None) ** Cell_PM)
        Observable('M_free', M(bRA=None, bSTAT1=None) ** Cell_CP)

        Observable('IL_RA1', IL(bA1=1, bA2=None, bB1=None, bB2=None, B_t='SB') ** EC % RA(bA=1, bM=None) ** Cell_PM)
        Observable('IL_RA2', IL(bA1=None, bA2=1, bB1=None, bB2=None, B_t='SB') ** EC % RA(bA=1, bM=None) ** Cell_PM)
        Observable('IL_RB1', IL(bA1=None, bA2=None, bB1=1, bB2=None, B_t='SB') ** EC % RB(bB=1) ** Cell_PM)
        Observable('IL_RB2', IL(bA1=None, bA2=None, bB1=None, bB2=1, B_t='SB') ** EC % RB(bB=1) ** Cell_PM)

        Observable('IL_RA1_RA2', IL(bA1=1, bA2=2, bB1=None, bB2=None, B_t='DB') ** EC % RA(bA=1, bM=None) ** Cell_PM % RA(bA=2, bM=None) ** Cell_PM)
        Observable('IL_RA1_RB1', IL(bA1=1, bA2=None, bB1=2, bB2=None, B_t='DB') ** EC % RA(bA=1, bM=None) ** Cell_PM % RB(bB=2) ** Cell_PM)
        Observable('IL_RA1_RB2', IL(bA1=1, bA2=None, bB1=None, bB2=2, B_t='DB') ** EC % RA(bA=1, bM=None) ** Cell_PM % RB(bB=2) ** Cell_PM)
        Observable('IL_RA2_RB1', IL(bA1=None, bA2=1, bB1=2, bB2=None, B_t='DB') ** EC % RA(bA=1, bM=None) ** Cell_PM % RB(bB=2) ** Cell_PM)
        Observable('IL_RA2_RB2', IL(bA1=None, bA2=1, bB1=None, bB2=2, B_t='DB') ** EC % RA(bA=1, bM=None) ** Cell_PM % RB(bB=2) ** Cell_PM)
        Observable('IL_RB1_RB2', IL(bA1=None, bA2=None, bB1=1, bB2=2, B_t='DB') ** EC % RB(bB=1) ** Cell_PM % RB(bB=2) ** Cell_PM)

        Observable('IL_RA1_RA2_RB1', IL(bA1=1, bA2=2, bB1=3, bB2=None, B_t='TB') ** EC % RA(bA=1, bM=None) ** Cell_PM % RA(bA=2, bM=None) ** Cell_PM % RB(bB=3) ** Cell_PM)
        Observable('IL_RA1_RA2_RB2', IL(bA1=1, bA2=2, bB1=None, bB2=3, B_t='TB') ** EC % RA(bA=1, bM=None) ** Cell_PM % RA(bA=2, bM=None) ** Cell_PM % RB(bB=3) ** Cell_PM)
        Observable('IL_RA1_RB1_RB2', IL(bA1=1, bA2=None, bB1=2, bB2=3, B_t='TB') ** EC % RA(bA=1, bM=None) ** Cell_PM % RB(bB=2) ** Cell_PM % RB(bB=3) ** Cell_PM)
        Observable('IL_RA2_RB1_RB2', IL(bA1=None, bA2=1, bB1=2, bB2=3, B_t='TB') ** EC % RA(bA=1, bM=None) ** Cell_PM % RB(bB=2) ** Cell_PM % RB(bB=3) ** Cell_PM)

        Observable('IL_RA1_RA2_RB1_RB2', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bM=None) ** Cell_PM % RA(bA=2, bM=None) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM)

        Observable('IL_RA1_RB1_P1', IL(bA1=1, bA2=None, bB1=2, bB2=None, B_t='DB') ** EC % RA(bA=1, bM=5) ** Cell_PM % RB(bB=2) ** Cell_PM % M(bRA=5, bSTAT1=None) ** Cell_CP)
        Observable('IL_RA1_RB1_PS1', IL(bA1=1, bA2=None, bB1=2, bB2=None, B_t='DB') ** EC % RA(bA=1, bM=5) ** Cell_PM % RB(bB=2) ** Cell_PM % M(bRA=5, bSTAT1=7) ** Cell_CP % STAT1(b1=7, P='u') ** Cell_CP)
        Observable('IL_RA2_RB2_P2', IL(bA1=None, bA2=1, bB1=None, bB2=2, B_t='DB') ** EC % RA(bA=1, bM=6) ** Cell_PM % RB(bB=2) ** Cell_PM % M(bRA=6, bSTAT1=None) ** Cell_CP)
        Observable('IL_RA2_RB2_PS2', IL(bA1=None, bA2=1, bB1=None, bB2=2, B_t='DB') ** EC % RA(bA=1, bM=6) ** Cell_PM % RB(bB=2) ** Cell_PM % M(bRA=6, bSTAT1=8) ** Cell_CP % STAT1(b1=8, P='u') ** Cell_CP)

        Observable('IL_RA1_RA2_RB1_P1', IL(bA1=1, bA2=2, bB1=3, bB2=None, B_t='TB') ** EC % RA(bA=1, bM=5) ** Cell_PM % RA(bA=2, bM=None) ** Cell_PM % RB(bB=3) ** Cell_PM % M(bRA=5, bSTAT1=None) ** Cell_CP)
        Observable('IL_RA1_RA2_RB1_PS1', IL(bA1=1, bA2=2, bB1=3, bB2=None, B_t='TB') ** EC % RA(bA=1, bM=5) ** Cell_PM % RA(bA=2, bM=None) ** Cell_PM % RB(bB=3) ** Cell_PM % M(bRA=5, bSTAT1=7) ** Cell_CP % STAT1(b1=7, P='u') ** Cell_CP)
        Observable('IL_RA1_RA2_RB2_P2', IL(bA1=1, bA2=2, bB1=None, bB2=3, B_t='TB') ** EC % RA(bA=1, bM=None) ** Cell_PM % RA(bA=2, bM=6) ** Cell_PM % RB(bB=3) ** Cell_PM % M(bRA=6, bSTAT1=None) ** Cell_CP)
        Observable('IL_RA1_RA2_RB2_PS2', IL(bA1=1, bA2=2, bB1=None, bB2=3, B_t='TB') ** EC % RA(bA=1, bM=None) ** Cell_PM % RA(bA=2, bM=6) ** Cell_PM % RB(bB=3) ** Cell_PM % M(bRA=6, bSTAT1=8) ** Cell_CP % STAT1(b1=8, P='u') ** Cell_CP)
        Observable('IL_RA1_RB1_RB2_P1', IL(bA1=1, bA2=None, bB1=2, bB2=3, B_t='TB') ** EC % RA(bA=1, bM=5) ** Cell_PM % RB(bB=2) ** Cell_PM % RB(bB=3) ** Cell_PM % M(bRA=5, bSTAT1=None) ** Cell_CP)
        Observable('IL_RA1_RB1_RB2_PS1', IL(bA1=1, bA2=None, bB1=2, bB2=3, B_t='TB') ** EC % RA(bA=1, bM=5) ** Cell_PM % RB(bB=2) ** Cell_PM % RB(bB=3) ** Cell_PM % M(bRA=5, bSTAT1=7) ** Cell_CP % STAT1(b1=7, P='u') ** Cell_CP)
        Observable('IL_RA2_RB1_RB2_P2', IL(bA1=None, bA2=1, bB1=2, bB2=3, B_t='TB') ** EC % RA(bA=1, bM=6) ** Cell_PM % RB(bB=2) ** Cell_PM % RB(bB=3) ** Cell_PM % M(bRA=6, bSTAT1=None) ** Cell_CP)
        Observable('IL_RA2_RB1_RB2_PS2', IL(bA1=None, bA2=1, bB1=2, bB2=3, B_t='TB') ** EC % RA(bA=1, bM=6) ** Cell_PM % RB(bB=2) ** Cell_PM % RB(bB=3) ** Cell_PM % M(bRA=6, bSTAT1=8) ** Cell_CP % STAT1(b1=8, P='u') ** Cell_CP)

        Observable('IL_RA1_RA2_RB1_RB2_P1', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bM=5) ** Cell_PM % RA(bA=2, bM=None) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % M(bRA=5, bSTAT1=None) ** Cell_CP)
        Observable('IL_RA1_RA2_RB1_RB2_PS1', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bM=5) ** Cell_PM % RA(bA=2, bM=None) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % M(bRA=5, bSTAT1=7) ** Cell_CP % STAT1(b1=7, P='u') ** Cell_CP)
        Observable('IL_RA1_RA2_RB1_RB2_P2', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bM=None) ** Cell_PM % RA(bA=2, bM=6) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % M(bRA=6, bSTAT1=None) ** Cell_CP)
        Observable('IL_RA1_RA2_RB1_RB2_PS2', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bM=None) ** Cell_PM % RA(bA=2, bM=6) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % M(bRA=6, bSTAT1=8) ** Cell_CP % STAT1(b1=8, P='u') ** Cell_CP)
        Observable('IL_RA1_RA2_RB1_RB2_P1_P2', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bM=5) ** Cell_PM % RA(bA=2, bM=6) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % M(bRA=5, bSTAT1=None) ** Cell_CP % M(bRA=6, bSTAT1=None) ** Cell_CP)
        Observable('IL_RA1_RA2_RB1_RB2_PS1_P2', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bM=5) ** Cell_PM % RA(bA=2, bM=6) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % M(bRA=5, bSTAT1=7) ** Cell_CP % M(bRA=6, bSTAT1=None) ** Cell_CP % STAT1(b1=7, P='u') ** Cell_CP)
        Observable('IL_RA1_RA2_RB1_RB2_P1_PS2', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bM=5) ** Cell_PM % RA(bA=2, bM=6) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % M(bRA=5, bSTAT1=None) ** Cell_CP % M(bRA=6, bSTAT1=8) ** Cell_CP % STAT1(b1=8, P='u') ** Cell_CP)
        Observable('IL_RA1_RA2_RB1_RB2_PS1_PS2', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bM=5) ** Cell_PM % RA(bA=2, bM=6) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % M(bRA=5, bSTAT1=7) ** Cell_CP % M(bRA=6, bSTAT1=8) ** Cell_CP % STAT1(b1=7, P='u') ** Cell_CP % STAT1(b1=8, P='u') ** Cell_CP)

        Observable('STAT1P', STAT1(b1=None, P='p') ** Cell_CP)
        
        dT, Tf = func_time(PA["IL0"].values[0])
        t = np.arange(0,Tf,dT/500)
        simulator = ScipyOdeSimulator(model, tspan=t, compiler='cython', integrator = 'lsoda').run().all
        PA["Result"] = simulator['STAT1P'][-1]
        
    return PA