from pysb import Model, Parameter, Compartment, Monomer, Parameter, Rule, Initial, Observable
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import cython
import matplotlib.pyplot as plt
import sympy as sp
import pickle
from collections import Counter
import sys, os
sys.setrecursionlimit(5000)
from src.PySBtoLF_func import generate_graphs,adapt_graph_get_rhos
from time import time

start_time = time()  # Record the start time
PA={
    "vol_EC" : 1.0E-10, # In dm # OG: 1.0E-10
    "surf_cell" : 1.11545244e-07, # OG: 1.4e-08 Mut_noSOCS: 1.4e-7
    "vol_cell" : 3.0E-12, # In dm
    "k_ILM_f" : 6.97E+05,
    "k_ILM_b" : 3.07214605e+00, #4.34 # OG: 4.34e-02 Mut_noSOCS: 8.34e-02
    "k_IL_RA_f" : 6.97E+05,
    "k_IL_RA_b" : 5.94742574e-04, # OG: 1.42E-04 Mut_noSOCS: 1.42E-03
    "k_IL_RB_f" : 6.97E+05,
    "k_IL_RB_b" : 1394, # OG: 1394
    "k_IL_RA_RB_f" : 6.97E+05,
    "k_IL_RA_RB_b" : 1.32365577e+00, # OG: 243.95
    "k_STAT1_f" : 6.55E+06*10,
    "k_STAT1_b" : 0.678,
    "k_M_f" : 6.55E+06*1000,
    "k_M_b" : 0.678,
    "k_STAT3_f" : 6.55E+06*1000,
    "k_STAT3_b" : 0.678,
    "k_PHOS" : 3.16E-03,
    "k_DEPHOS" : 0.000664996232095083
}

IC = {
    "IL0":1e-7,
    "RA0":750,
    "RB0":2323,
    "STAT10" : 477858.2479731492,
    "M0" : 477858.2479731492,
    "STAT30" : 477858.2479731492
    }

Na=6.023e23
Width_PM = 1e-7 # In dm
# Initialize model
Model()

# Compartment
Compartment(name='EC', parent=None, dimension=3, size=None)
Compartment(name='Cell_PM', parent=EC, dimension=2, size=None)
Compartment(name='Cell_CP', parent=Cell_PM, dimension=3, size=None)

# Monomers
Monomer('IL', ['bA1', 'bA2', 'bB1', 'bB2'])
Monomer('RA', ['bA', 'bMSTAT1'])
Monomer('RB', ['bB'])
Monomer('M', ['bRA','bSTAT1'])
Monomer('STAT1', ['b1','P'], {'P': ['u', 'p']})

# Parameters
Parameter('K_IL_RA_f', 2*PA["k_IL_RA_f"]/(Na*PA["vol_EC"]))
Parameter('K_IL_RA_b', PA["k_IL_RA_b"])
Parameter('K_IL_RB_f', 2*PA["k_IL_RB_f"]/(Na*PA["vol_EC"]))
Parameter('K_IL_RB_b', PA["k_IL_RB_b"])

Parameter('K_IL_RA2_f', PA["k_IL_RA_f"]/(Na*PA["surf_cell"]*Width_PM))
Parameter('K_IL_RA2_b', PA["k_IL_RA_b"])
Parameter('K_IL_RB2_f', PA["k_IL_RB_f"]/(Na*PA["surf_cell"]*Width_PM))
Parameter('K_IL_RB2_b', PA["k_IL_RB_b"])
Parameter('K_IL_RA_RB_f', PA["k_IL_RA_RB_f"]/(Na*PA["surf_cell"]*Width_PM))
Parameter('K_IL_RA_RB_b', PA["k_IL_RA_RB_b"])

Parameter('K_IL_RA2_2_f', PA["k_IL_RA_f"]/(Na*PA["surf_cell"]*Width_PM)*(1/2))
Parameter('K_IL_RB2_2_f', PA["k_IL_RB_f"]/(Na*PA["surf_cell"]*Width_PM)*(1/2))

Parameter('K_IL_RA2_3_b', PA["k_IL_RA_b"]*(1/2))
Parameter('K_IL_RA_RB_3_b', PA["k_IL_RA_RB_b"]*(1/2))

Parameter('K_IL_RA2_4_b', PA["k_IL_RA_b"]*2)
Parameter('K_IL_RA_RB_4_b', PA["k_IL_RA_RB_b"]*2)

Parameter('K_M_f', PA["k_M_f"]/(Na*PA["vol_cell"]))
Parameter('K_M_b', PA["k_M_b"])
Parameter('K_M_2_f', PA["k_M_f"]/(Na*PA["vol_cell"])*2)
Parameter('K_M_3_f', PA["k_M_f"]/(Na*PA["vol_cell"])*(1/2))

Parameter('K_STAT1_f', PA["k_STAT1_f"]/(Na*PA["vol_cell"]))
Parameter('K_STAT1_b', PA["k_STAT1_b"])
Parameter('K_STAT1_2_f', PA["k_STAT1_f"]/(Na*PA["vol_cell"])*2)
Parameter('K_STAT1_3_f', PA["k_STAT1_f"]/(Na*PA["vol_cell"])*(1/2))

# Complexes and reactions (binding of receptors)
IL_free = IL(bA1=None, bA2=None, bB1=None, bB2=None)
IL_RA1_alone = IL(bA1=1, bA2=None, bB1=None, bB2=None) % RA(bA=1, bMSTAT1=None)
Rule('IL_RA1_alone_binding', IL_free + RA(bA=None, bMSTAT1=None)| IL_RA1_alone, *[K_IL_RA_f, K_IL_RA_b])

# Binding of second receptors from IL-RA1 complex
IL_RA1_RA2 = IL(bA1=1, bA2=2, bB1=None, bB2=None) % RA(bA=1, bMSTAT1=None) % RA(bA=2, bMSTAT1=None)
Rule('IL_RA1_RA2_binding_DB', IL_RA1_alone + RA(bA=None, bMSTAT1=None) | IL_RA1_RA2, *[K_IL_RA2_2_f, K_IL_RA2_b])
IL_RA1_RB1 = IL(bA1=1, bA2=None, bB1=2, bB2=None) % RA(bA=1, bMSTAT1=None) % RB(bB=2)
Rule('IL_RA1_RB1_binding_DB', IL_RA1_alone + RB(bB=None) | IL_RA1_RB1, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
IL_RA1_RB1_M1 = IL(bA1=1, bA2=None, bB1=2, bB2=None) % RA(bA=1, bMSTAT1=5) % RB(bB=2) % M(bRA=5, bSTAT1=None) 
IL_RA1_RB1_MS1 = IL(bA1=1, bA2=None, bB1=2, bB2=None) % RA(bA=1, bMSTAT1=5) % RB(bB=2) % M(bRA=5, bSTAT1=6) % STAT1(b1=6, P='u')
Rule('IL_RA1_RB1_unbinding_DB', IL_RA1_RB1_M1 >> IL_RA1_alone + RB(bB=None) + M(bRA=None, bSTAT1=None) , K_IL_RA_RB_b)
Rule('IL_RA1_RB1_unbinding_DB2', IL_RA1_RB1_MS1 >> IL_RA1_alone + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_b)

# Binding of third receptors to form IL-RA1-RA2-RB1 complex
IL_RA1_RA2_RB1 = IL(bA1=1, bA2=2, bB1=3, bB2=None) % RA(bA=1, bMSTAT1=None) % RA(bA=2, bMSTAT1=None) % RB(bB=3)
IL_RA1_RA2_RB1_M1 = IL(bA1=1, bA2=2, bB1=3, bB2=None) % RA(bA=1, bMSTAT1=5) % RA(bA=2, bMSTAT1=None) % RB(bB=3) % M(bRA=5, bSTAT1=None)
IL_RA1_RA2_RB1_MS1 = IL(bA1=1, bA2=2, bB1=3, bB2=None) % RA(bA=1, bMSTAT1=5) % RA(bA=2, bMSTAT1=None) % RB(bB=3) % M(bRA=5, bSTAT1=6) % STAT1(b1=6, P='u')

Rule('IL_RA1RA1_RB1_binding_TB', IL_RA1_RA2 + RB(bB=None) | IL_RA1_RA2_RB1, *[K_IL_RA_RB_f, K_IL_RA_RB_3_b])
Rule('IL_RA1RA1_RB1_unbinding_TB', IL_RA1_RA2_RB1_M1 >> IL_RA1_RA2 + RB(bB=None) + M(bRA=None, bSTAT1=None), K_IL_RA_RB_3_b)
Rule('IL_RA1RA1_RB1_unbinding_TB2', IL_RA1_RA2_RB1_MS1 >> IL_RA1_RA2 + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_3_b)

Rule('IL_RA1RB1_RA2_binding_TB', IL_RA1_RB1 + RA(bA=None, bMSTAT1=None) | IL_RA1_RA2_RB1, *[K_IL_RA2_f, K_IL_RA2_b])
Rule('IL_RA1RB1_RA2_binding_TB2', IL_RA1_RB1_M1 + RA(bA=None, bMSTAT1=None) | IL_RA1_RA2_RB1_M1, *[K_IL_RA2_f, K_IL_RA2_b])
Rule('IL_RA1RB1_RA2_binding_TB3', IL_RA1_RB1_MS1 + RA(bA=None, bMSTAT1=None) | IL_RA1_RA2_RB1_MS1, *[K_IL_RA2_f, K_IL_RA2_b])

       
# Binding of fourth receptor
IL_RA1_RA2_RB1_RB2 = IL(bA1=1, bA2=2, bB1=3, bB2=4) % RA(bA=1, bMSTAT1=None) % RA(bA=2, bMSTAT1=None) % RB(bB=3) % RB(bB=4)
IL_RA1_RA2_RB1_RB2_M1 = IL(bA1=1, bA2=2, bB1=3, bB2=4) % RA(bA=1, bMSTAT1=5) % RA(bA=2, bMSTAT1=None) % RB(bB=3) % RB(bB=4) % M(bRA=5, bSTAT1=None)
IL_RA1_RA2_RB1_RB2_MS1 = IL(bA1=1, bA2=2, bB1=3, bB2=4) % RA(bA=1, bMSTAT1=5) % RA(bA=2, bMSTAT1=None) % RB(bB=3) % RB(bB=4) % M(bRA=5, bSTAT1=6) % STAT1(b1=6, P='u')
IL_RA1_RA2_RB1_RB2_MS1_M2 = IL(bA1=1, bA2=2, bB1=3, bB2=4) % RA(bA=1, bMSTAT1=5) % RA(bA=2, bMSTAT1=7) % RB(bB=3) % RB(bB=4) % M(bRA=5, bSTAT1=6) % STAT1(b1=6, P='u') % M(bRA=7, bSTAT1=None)
IL_RA1_RA2_RB1_RB2_MS1_MS2 = IL(bA1=1, bA2=2, bB1=3, bB2=4) % RA(bA=1, bMSTAT1=5) % RA(bA=2, bMSTAT1=7) % RB(bB=3) % RB(bB=4) % M(bRA=5, bSTAT1=6) % STAT1(b1=6, P='u') % M(bRA=7, bSTAT1=8) % STAT1(b1=8, P='u')
    
Rule('IL_RA1RA2RB1_RB2_binding_QB', IL_RA1_RA2_RB1 + RB(bB=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA_RB_f, K_IL_RA_RB_4_b])
Rule('IL_RA1RA2RB1_RB2_M1_binding_QB', IL_RA1_RA2_RB1_M1 + RB(bB=None) | IL_RA1_RA2_RB1_RB2_M1, *[K_IL_RA_RB_f, K_IL_RA_RB_4_b])
Rule('IL_RA1RA2RB1_RB2_M1_unbinding_QB', IL_RA1_RA2_RB1_RB2_M1 >> IL_RA1_RA2_RB1 + RB(bB=None) + M(bRA=None, bSTAT1=None), K_IL_RA_RB_4_b)
Rule('IL_RA1RA2RB1_RB2_MS1_binding_QB', IL_RA1_RA2_RB1_MS1 + RB(bB=None) | IL_RA1_RA2_RB1_RB2_MS1, *[K_IL_RA_RB_f, K_IL_RA_RB_4_b])
Rule('IL_RA1RA2RB1_RB2_MS1_unbinding_QB', IL_RA1_RA2_RB1_RB2_MS1 >> IL_RA1_RA2_RB1 + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_4_b)
Rule('IL_RA1RA2RB1_RB2_MS1_M2_unbinding_QB1', IL_RA1_RA2_RB1_RB2_MS1_M2 >> IL_RA1_RA2_RB1_M1 + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_4_b)
Rule('IL_RA1RA2RB1_RB2_MS1_M2_unbinding_QB2', IL_RA1_RA2_RB1_RB2_MS1_M2 >> IL_RA1_RA2_RB1_MS1 + RB(bB=None) + M(bRA=None, bSTAT1=None), K_IL_RA_RB_4_b)
Rule('IL_RA1RA2RB1_RB2_MS1_MS2_unbinding_QB', IL_RA1_RA2_RB1_RB2_MS1_MS2 >> IL_RA1_RA2_RB1_MS1 + RB(bB=None) + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_IL_RA_RB_4_b)


# Binding of M and STAT
Rule('IL_RA1_RB1_M1_binding', IL_RA1_RB1 + M(bRA=None, bSTAT1=None) | IL_RA1_RB1_M1, *[K_M_f, K_M_b])
Rule('IL_RA1_RB1_MS1_binding', IL_RA1_RB1_M1 + STAT1(b1=None, P='u') | IL_RA1_RB1_MS1, *[K_STAT1_f, K_STAT1_b])
Rule('IL_RA1_RB1_MS1_unbinding', IL_RA1_RB1_MS1 >> IL_RA1_RB1 + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_M_b)

Rule('IL_RA1_RA2_RB1_M1_binding', IL_RA1_RA2_RB1 + M(bRA=None, bSTAT1=None) | IL_RA1_RA2_RB1_M1, *[K_M_f, K_M_b])
Rule('IL_RA1_RA2_RB1_MS1_binding', IL_RA1_RA2_RB1_M1 + STAT1(b1=None, P='u') | IL_RA1_RA2_RB1_MS1, *[K_STAT1_f, K_STAT1_b])
Rule('IL_RA1_RA2_RB1_MS1_unbinding', IL_RA1_RA2_RB1_MS1 >> IL_RA1_RA2_RB1 + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_M_b)

Rule('IL_IL_RA1_RA2_RB1_RB2_M1_binding', IL_RA1_RA2_RB1_RB2 + M(bRA=None, bSTAT1=None) | IL_RA1_RA2_RB1_RB2_M1, *[K_M_2_f, K_M_b])
Rule('IL_IL_RA1_RA2_RB1_RB2_MS1_binding', IL_RA1_RA2_RB1_RB2_M1 + STAT1(b1=None, P='u') | IL_RA1_RA2_RB1_RB2_MS1, *[K_STAT1_f, K_STAT1_b])
Rule('IL_IL_RA1_RA2_RB1_RB2_MS1_unbinding', IL_RA1_RA2_RB1_RB2_MS1 >> IL_RA1_RA2_RB1_RB2 + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_M_b)
Rule('IL_IL_RA1_RA2_RB1_RB2_MS1_M2_binding1', IL_RA1_RA2_RB1_RB2_MS1 + M(bRA=None, bSTAT1=None) | IL_RA1_RA2_RB1_RB2_MS1_M2, *[K_M_3_f, K_M_b])
Rule('IL_IL_RA1_RA2_RB1_RB2_MS1_M2_unbinding', IL_RA1_RA2_RB1_RB2_MS1_M2 >> IL_RA1_RA2_RB1_RB2_M1 + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_M_b)
Rule('IL_IL_RA1_RA2_RB1_RB2_MS1_MS2_binding', IL_RA1_RA2_RB1_RB2_MS1_M2 + STAT1(b1=None, P='u') | IL_RA1_RA2_RB1_RB2_MS1_MS2, *[K_STAT1_3_f, K_STAT1_b])
Rule('IL_IL_RA1_RA2_RB1_RB2_MS1_MS2_unbinding', IL_RA1_RA2_RB1_RB2_MS1_MS2 >> IL_RA1_RA2_RB1_RB2_MS1 + M(bRA=None, bSTAT1=None) + STAT1(b1=None, P='u'), K_M_b)

# Initial conditions
Parameter('ILD_0', IC["IL0"]*Na*PA["vol_EC"])
Initial(IL(bA1=None, bA2=None, bB1=None, bB2=None) ** EC, ILD_0)

Parameter('RA_0',IC["RA0"])
Parameter('RB_0',IC["RB0"])
Initial(RA(bA=None, bMSTAT1=None) ** Cell_PM, RA_0)
Initial(RB(bB=None) ** Cell_PM, RB_0)

Parameter('STAT_1_0', IC["STAT10"])
Initial(STAT1(b1=None, P='u') ** Cell_CP, STAT_1_0)
Parameter('M_0', IC["M0"])
Initial(M(bRA=None,bSTAT1=None) ** Cell_CP, M_0)

# Simulation
Observable('IL_free', IL(bA1=None, bA2=None, bB1=None, bB2=None) ** EC)
Observable('RA_free', RA(bA=None, bMSTAT1=None) ** Cell_PM)
Observable('RB_free', RB(bB=None) ** Cell_PM)
Observable('S1_free', STAT1(b1=None, P='u') ** Cell_CP)

Observable('IL_RA1', IL(bA1=1, bA2=None, bB1=None, bB2=None) ** EC % RA(bA=1, bMSTAT1=None) ** Cell_PM)

Observable('IL_RA1_RA2', IL(bA1=1, bA2=2, bB1=None, bB2=None) ** EC % RA(bA=1, bMSTAT1=None) ** Cell_PM % RA(bA=2, bMSTAT1=None) ** Cell_PM)
Observable('IL_RA1_RB1', IL(bA1=1, bA2=None, bB1=2, bB2=None) ** EC % RA(bA=1, bMSTAT1=None) ** Cell_PM % RB(bB=2) ** Cell_PM)

Observable('IL_RA1_RA2_RB1', IL(bA1=1, bA2=2, bB1=3, bB2=None) ** EC % RA(bA=1, bMSTAT1=None) ** Cell_PM % RA(bA=2, bMSTAT1=None) ** Cell_PM % RB(bB=3) ** Cell_PM)

Observable('IL_RA1_RA2_RB1_RB2', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bMSTAT1=None) ** Cell_PM % RA(bA=2, bMSTAT1=None) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM)

Observable('IL_RA1_RB1_P1', IL(bA1=1, bA2=None, bB1=2, bB2=None) ** EC % RA(bA=1, bMSTAT1=5) ** Cell_PM % RB(bB=2) ** Cell_PM % M(bRA=5, bSTAT1=None) ** Cell_CP)
Observable('IL_RA1_RB1_PS1', IL(bA1=1, bA2=None, bB1=2, bB2=None) ** EC % RA(bA=1, bMSTAT1=5) ** Cell_PM % RB(bB=2) ** Cell_PM % M(bRA=5, bSTAT1=6) ** Cell_CP % STAT1(b1=6, P='u') ** Cell_CP)

Observable('IL_RA1_RA2_RB1_P1', IL(bA1=1, bA2=2, bB1=3, bB2=None) ** EC % RA(bA=1, bMSTAT1=5) ** Cell_PM % RA(bA=2, bMSTAT1=None) ** Cell_PM % RB(bB=3) ** Cell_PM % M(bRA=5, bSTAT1=None) ** Cell_CP)
Observable('IL_RA1_RA2_RB1_PS1', IL(bA1=1, bA2=2, bB1=3, bB2=None) ** EC % RA(bA=1, bMSTAT1=5) ** Cell_PM % RA(bA=2, bMSTAT1=None) ** Cell_PM % RB(bB=3) ** Cell_PM % M(bRA=5, bSTAT1=6) ** Cell_CP % STAT1(b1=6, P='u') ** Cell_CP)
    
Observable('IL_RA1_RA2_RB1_RB2_P1', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bMSTAT1=5) ** Cell_PM % RA(bA=2, bMSTAT1=None) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % M(bRA=5, bSTAT1=None) ** Cell_CP)
Observable('IL_RA1_RA2_RB1_RB2_PS1', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bMSTAT1=5) ** Cell_PM % RA(bA=2, bMSTAT1=None) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % M(bRA=5, bSTAT1=6) ** Cell_CP % STAT1(b1=6, P='u') ** Cell_CP)
Observable('IL_RA1_RA2_RB1_RB2_PS1_P2', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bMSTAT1=5) ** Cell_PM % RA(bA=2, bMSTAT1=7) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % M(bRA=5, bSTAT1=6) ** Cell_CP % M(bRA=7, bSTAT1=None) ** Cell_CP % STAT1(b1=6, P='u') ** Cell_CP)
Observable('IL_RA1_RA2_RB1_RB2_PS1_PS2', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bMSTAT1=5) ** Cell_PM % RA(bA=2, bMSTAT1=7) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % M(bRA=5, bSTAT1=6) ** Cell_CP % M(bRA=7, bSTAT1=8) ** Cell_CP % STAT1(b1=6, P='u') ** Cell_CP % STAT1(b1=8, P='u') ** Cell_CP)

t = np.arange(0,1*5,1/100)
simulator = ScipyOdeSimulator(model, tspan=t, compiler='python', integrator = 'lsoda').run()

graph_gen_species = model.species[:5] # model.species[1:4] -> for the model of dimeric IL10, model.species[:3] -> for model monomeric IL10
graphs_system = generate_graphs(model, graph_gen_species) # Get graphs of the system (not fully adapted for rhos script)

species_ggs_new = [sp.Symbol("IL10D"),sp.Symbol("RA"),sp.Symbol("RB"),sp.Symbol("S1"),sp.Symbol("M")] # Get new symbols for IC (graph generating species or monomers
graph_final,new_index_edges,new_index_species,dict_repeated_weights_names = adapt_graph_get_rhos(model, graphs_system, species_ggs_new, sp.Symbol("__s0"), graph_gen_species) # Adapt graph for rhos script (Put __s1 as graph generaitng species is the IL10 dimer)

graph_final.append((1,'K_ILM_b',15))
graph_final.append((15,'K_ILM_f-IL10M',1))
new_index_species[14] = "ILM() ** EC"
species_ggs_new = species_ggs_new + [sp.Symbol("IL10M")] # Add the monomer species

print("Number of nodes: "+str(len(graph_final)))
print("Number of edges: "+str(len(new_index_species)))

# Paths to dependancies (git repositories)
path_to_eigen="/home/qmarti/Documents/repos/eigen/"
path_to_polynomials="\"/home/qmarti/Documents/repos/polynomials/include/polynomial/\""
path_to_utilsGRF="\"/home/qmarti/Documents/repos/GeneRegulatoryFunctions/utilsGRF\"" #GeneRegulatoryFunctions repo
path_to_utilsGRF2='/home/qmarti/Documents/repos/GeneRegulatoryFunctions/utilsGRF' #for python, the space is a problem between python and bash
MTTfolder='/home/qmarti/Documents/repos/GeneRegulatoryFunctions/utilsGRF' #folder with MTT.py 
basename='G_IL'

sys.path.append(path_to_utilsGRF2) #this is the GeneRegulatoryFunctions repo 
import writescripts

# Get parameters of the graph
edges = graph_final
parlist=[x[1] for x in edges]
for pnum,par in enumerate(parlist):
    for species in species_ggs_new: 
        if '-'+str(species) in par:
            parlist[pnum]=parlist[pnum].replace('-'+str(species),'')
parlist = list(set(parlist))

# Define PrepareFiles class to make calculations for graphs. As we have directly defined the 
obj=writescripts.PrepareFilesNoneq(edgelist=graph_final,varGRF='IL10D',concvars=['IL10M','RA','RB','S1','M'],parlist=parlist,MTTfolder=MTTfolder,graphbasename=basename)# Gets all spanning trees for each node and gets the correpondant rho
obj.write_execute_parse()
# Simplify rho
#obj.simpify_rhos()

# Revert changes to the parameter names so there are no repeated edge weights
rho_list = []
for rho in obj.all_rhos:
    for weight in dict_repeated_weights_names.keys():
        rho = rho.replace(weight,dict_repeated_weights_names[weight])
    rho_list.append(rho)

with open("rhos_STAT1_simpl.pkl", "wb") as file:
    pickle.dump(rho_list, file)

end_time = time()  # Record the end time
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")