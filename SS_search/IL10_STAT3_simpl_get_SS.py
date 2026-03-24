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
Monomer('RA', ['bA', 'bSTAT3'])
Monomer('RB', ['bB'])
Monomer('STAT3', ['b3','P'], {'P': ['u', 'p']})


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

Parameter('K_STAT3_f', PA["k_STAT3_f"]/(Na*PA["vol_cell"]))
Parameter('K_STAT3_b', PA["k_STAT3_b"])

Parameter('K_STAT3_2_f', PA["k_STAT3_f"]/(Na*PA["vol_cell"])*2)
Parameter('K_STAT3_3_f', PA["k_STAT3_f"]/(Na*PA["vol_cell"])*(1/2))


# Complexes and reactions (binding of receptors)
IL_free = IL(bA1=None, bA2=None, bB1=None, bB2=None)
IL_RA1_alone = IL(bA1=1, bA2=None, bB1=None, bB2=None) % RA(bA=1, bSTAT3=None)
Rule('IL_RA1_alone_binding', IL_free + RA(bA=None, bSTAT3=None)| IL_RA1_alone, *[K_IL_RA_f, K_IL_RA_b])

# Binding of second receptors from IL-RA1 complex
IL_RA1_RA2 = IL(bA1=1, bA2=2, bB1=None, bB2=None) % RA(bA=1, bSTAT3=None) % RA(bA=2, bSTAT3=None)
Rule('IL_RA1_RA2_binding_DB', IL_RA1_alone + RA(bA=None, bSTAT3=None) | IL_RA1_RA2, *[K_IL_RA2_2_f, K_IL_RA2_b])
IL_RA1_RB1 = IL(bA1=1, bA2=None, bB1=2, bB2=None) % RA(bA=1, bSTAT3=None) % RB(bB=2)
Rule('IL_RA1_RB1_binding_DB', IL_RA1_alone + RB(bB=None) | IL_RA1_RB1, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
IL_RA1_RB1_S31 = IL(bA1=1, bA2=None, bB1=2, bB2=None) % RA(bA=1, bSTAT3=5) % RB(bB=2) % STAT3(b3=5, P='u')
Rule('IL_RA1_RB1_unbinding_DB', IL_RA1_RB1_S31 >> IL_RA1_alone + RB(bB=None) + STAT3(b3=None, P='u'), K_IL_RA_RB_b)

# Binding of third receptors to form IL-RA1-RA2-RB1 complex
IL_RA1_RA2_RB1 = IL(bA1=1, bA2=2, bB1=3, bB2=None) % RA(bA=1, bSTAT3=None) % RA(bA=2, bSTAT3=None) % RB(bB=3)
Rule('IL_RA1RA1_RB1_binding_TB', IL_RA1_RA2 + RB(bB=None) | IL_RA1_RA2_RB1, *[K_IL_RA_RB_f, K_IL_RA_RB_3_b])
IL_RA1_RA2_RB1_S31 = IL(bA1=1, bA2=2, bB1=3, bB2=None) % RA(bA=1, bSTAT3=5) % RA(bA=2, bSTAT3=None) % RB(bB=3) % STAT3(b3=5, P='u')
Rule('IL_RA1RA1_RB1_unbinding_TB', IL_RA1_RA2_RB1_S31 >> IL_RA1_RA2 + RB(bB=None) + STAT3(b3=None, P='u'), K_IL_RA_RB_3_b)
Rule('IL_RA1RB1_RA2_binding_TB', IL_RA1_RB1 + RA(bA=None, bSTAT3=None) | IL_RA1_RA2_RB1, *[K_IL_RA2_f, K_IL_RA2_b])
Rule('IL_RA1RB1_RA2_binding_TB2', IL_RA1_RB1_S31 + RA(bA=None, bSTAT3=None) | IL_RA1_RA2_RB1_S31, *[K_IL_RA2_f, K_IL_RA2_b])


# Binding of fourth receptor
IL_RA1_RA2_RB1_RB2 = IL(bA1=1, bA2=2, bB1=3, bB2=4) % RA(bA=1, bSTAT3=None) % RA(bA=2, bSTAT3=None) % RB(bB=3) % RB(bB=4)
IL_RA1_RA2_RB1_RB2_S31 = IL(bA1=1, bA2=2, bB1=3, bB2=4) % RA(bA=1, bSTAT3=5) % RA(bA=2, bSTAT3=None) % RB(bB=3) % RB(bB=4) % STAT3(b3=5, P='u')
IL_RA1_RA2_RB1_RB2_S31_S32 = IL(bA1=1, bA2=2, bB1=3, bB2=4) % RA(bA=1, bSTAT3=5) % RA(bA=2, bSTAT3=6) % RB(bB=3) % RB(bB=4) % STAT3(b3=5, P='u') % STAT3(b3=6, P='u')

Rule('IL_RA1RA2RB1_RB2_binding_QB', IL_RA1_RA2_RB1 + RB(bB=None) | IL_RA1_RA2_RB1_RB2, *[K_IL_RA_RB_f, K_IL_RA_RB_4_b])
Rule('IL_RA1RA2RB1_RB2_binding_QB2', IL_RA1_RA2_RB1_S31 + RB(bB=None) | IL_RA1_RA2_RB1_RB2_S31, *[K_IL_RA_RB_f, K_IL_RA_RB_4_b])
Rule('IL_RA1RA2RB1_RB2_unbinding_QB', IL_RA1_RA2_RB1_RB2_S31_S32 >> IL_RA1_RA2_RB1_S31 + RB(bB=None) + STAT3(b3=None, P='u'), K_IL_RA_RB_4_b)
Rule('IL_RA1RA2RB1_RB2_unbinding_QB2', IL_RA1_RA2_RB1_RB2_S31 >> IL_RA1_RA2_RB1 + RB(bB=None) + STAT3(b3=None, P='u'), K_IL_RA_RB_4_b) # Maybe

# Binding of STAT
Rule('IL_RA1_RB1_S3_binding', IL_RA1_RB1 + STAT3(b3=None, P='u') | IL_RA1_RB1_S31, *[K_STAT3_f, K_STAT3_b])
Rule('IL_IL_RA1_RA2_RB1_S3_binding', IL_RA1_RA2_RB1 + STAT3(b3=None, P='u') | IL_RA1_RA2_RB1_S31, *[K_STAT3_f, K_STAT3_b])
Rule('IL_IL_RA1_RA2_RB1_RB2_S31_binding', IL_RA1_RA2_RB1_RB2 + STAT3(b3=None, P='u') | IL_RA1_RA2_RB1_RB2_S31, *[K_STAT3_2_f, K_STAT3_b])
Rule('IL_IL_RA1_RA2_RB1_RB2_S31_S32_binding', IL_RA1_RA2_RB1_RB2_S31 + STAT3(b3=None, P='u') | IL_RA1_RA2_RB1_RB2_S31_S32, *[K_STAT3_3_f, K_STAT3_b])

# Initial conditions
Parameter('ILD_0', IC["IL0"]*Na*PA["vol_EC"])
Initial(IL(bA1=None, bA2=None, bB1=None, bB2=None) ** EC, ILD_0)

Parameter('RA_0',IC["RA0"])
Parameter('RB_0',IC["RB0"])
Initial(RA(bA=None, bSTAT3=None) ** Cell_PM, RA_0)
Initial(RB(bB=None) ** Cell_PM, RB_0)

Parameter('STAT_3_0', IC["STAT30"])
Initial(STAT3(b3=None, P='u') ** Cell_CP, STAT_3_0)

# Simulation
Observable('IL_free', IL(bA1=None, bA2=None, bB1=None, bB2=None) ** EC)
Observable('RA_free', RA(bA=None, bSTAT3=None) ** Cell_PM)
Observable('RB_free', RB(bB=None) ** Cell_PM)
Observable('S3_free', STAT3(b3=None, P='u') ** Cell_CP)

Observable('IL_RA1', IL(bA1=1, bA2=None, bB1=None, bB2=None) ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM)
Observable('IL_RB1', IL(bA1=None, bA2=None, bB1=1, bB2=None) ** EC % RB(bB=1) ** Cell_PM)

Observable('IL_RA1_RA2', IL(bA1=1, bA2=2, bB1=None, bB2=None) ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RA(bA=2, bSTAT3=None) ** Cell_PM)
Observable('IL_RA1_RB1', IL(bA1=1, bA2=None, bB1=2, bB2=None) ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RB(bB=2) ** Cell_PM)
Observable('IL_RA1_RB2', IL(bA1=1, bA2=None, bB1=None, bB2=2) ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RB(bB=2) ** Cell_PM)
Observable('IL_RB1_RB2', IL(bA1=None, bA2=None, bB1=1, bB2=2) ** EC % RB(bB=1) ** Cell_PM % RB(bB=2) ** Cell_PM)

Observable('IL_RA1_RA2_RB1', IL(bA1=1, bA2=2, bB1=3, bB2=None) ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RA(bA=2, bSTAT3=None) ** Cell_PM % RB(bB=3) ** Cell_PM)

Observable('IL_RA1_RA2_RB1_RB2', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bSTAT3=None) ** Cell_PM % RA(bA=2, bSTAT3=None) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM)

Observable('IL_RA1_RB1_S31', IL(bA1=1, bA2=None, bB1=2, bB2=None) ** EC % RA(bA=1, bSTAT3=5) ** Cell_PM % RB(bB=2) ** Cell_PM % STAT3(b3=5, P='u') ** Cell_CP)

Observable('IL_RA1_RA2_RB1_S31', IL(bA1=1, bA2=2, bB1=3, bB2=None) ** EC % RA(bA=1, bSTAT3=5) ** Cell_PM % RA(bA=2, bSTAT3=None) ** Cell_PM % RB(bB=3) ** Cell_PM % STAT3(b3=5, P='u') ** Cell_CP)

Observable('IL_RA1_RA2_RB1_RB2_S31', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bSTAT3=5) ** Cell_PM % RA(bA=2, bSTAT3=None) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % STAT3(b3=5, P='u') ** Cell_CP)
Observable('IL_RA1_RA2_RB1_RB2_S31_S32', IL(bA1=1, bA2=2, bB1=3, bB2=4) ** EC % RA(bA=1, bSTAT3=5) ** Cell_PM % RA(bA=2, bSTAT3=6) ** Cell_PM % RB(bB=3) ** Cell_PM % RB(bB=4) ** Cell_PM % STAT3(b3=5, P='u') ** Cell_CP % STAT3(b3=6, P='u') ** Cell_CP)

Observable('STAT3P', STAT3(b3=None, P='p') ** Cell_CP)

t = np.arange(0,1*5,1/100)
simulator = ScipyOdeSimulator(model, tspan=t, compiler='python', integrator = 'lsoda').run()

graph_gen_species = model.species[:4] # model.species[1:4] -> for the model of dimeric IL10, model.species[:3] -> for model monomeric IL10
graphs_system = generate_graphs(model, graph_gen_species) # Get graphs of the system (not fully adapted for rhos script)

species_ggs_new = [sp.Symbol("IL10D"),sp.Symbol("RA"),sp.Symbol("RB"),sp.Symbol("S3")] # Get new symbols for IC (graph generating species or monomers
graph_final,new_index_edges,new_index_species,dict_repeated_weights_names = adapt_graph_get_rhos(model, graphs_system, species_ggs_new, sp.Symbol("__s0"), graph_gen_species) # Adapt graph for rhos script (Put __s1 as graph generaitng species is the IL10 dimer)
graph_final.append((1,'K_ILM_b',11))
graph_final.append((11,'K_ILM_f-IL10M',1))
new_index_species[10] = "ILM() ** EC"
species_ggs_new = species_ggs_new + [sp.Symbol("IL10M")] # Add the monomer species

print("Number of nodes: "+str(len(graph_final)))
print("Number of edges: "+str(len(new_index_species)))

with open("rhos_STAT3_simpl.pkl", "rb") as file:
   rho_list = pickle.load(file)
    
parameters_change = {
    'K_ILM_f': "K_ILM_f*2",
    'K_ILM_b': "K_ILM_b*2",
    'K_IL_RA_f': "K_IL_RA_f*2",
    'K_IL_RB_f': "K_IL_RB_f*2",
    'K_IL_RA2_2_f': "K_IL_RA2_f*(1/2)",
    'K_IL_RB2_2_f': "K_IL_RB2_f*(1/2)",
    'K_IL_RA2_3_b': "K_IL_RA2_b*(1/2)",
    'K_IL_RA_RB_3_b': "K_IL_RA_RB_b*(1/2)",
    'K_IL_RA2_4_b': "K_IL_RA2_b*2",
    'K_IL_RA_RB_4_b': "K_IL_RA_RB_b*2",
    'K_STAT3_2_f': 'K_STAT3_f*2',
    'K_STAT3_3_f': 'K_STAT3_f*(1/2)'
}

for i in range(0,len(rho_list)):
    for param in parameters_change.keys():
        rho_list[i] = rho_list[i].replace(param,parameters_change[param])

rho_1 = sp.simplify(rho_list[0].split("=")[1]) # IL_free
rho_2 = sp.simplify(rho_list[1].split("=")[1]) # ILRA1
rho_3 = sp.simplify(rho_list[2].split("=")[1]) # ILRA1RA2
rho_4 = sp.simplify(rho_list[3].split("=")[1]) # ILRA1RB1
rho_5 = sp.simplify(rho_list[4].split("=")[1]) # ILRA1RA2RB1
rho_6 = sp.simplify(rho_list[5].split("=")[1]) # ILRA1RB1S31
rho_7 = sp.simplify(rho_list[6].split("=")[1]) # ILRA1RA2RB1S31
rho_8 = sp.simplify(rho_list[7].split("=")[1]) # ILRA1RA2RB1RB2
rho_9 = sp.simplify(rho_list[8].split("=")[1]) # ILRA1RA2RB1RB2S31
rho_10 = sp.simplify(rho_list[9].split("=")[1]) # ILRA1RA2RB1RB2S31S32
rho_11 = sp.simplify(rho_list[10].split("=")[1]) # ILM_free
rho_t = rho_1+rho_2+rho_3+rho_4+rho_5+rho_6+rho_7+rho_8+rho_9+rho_10+rho_11

IL10M0 = sp.Symbol("IL10M0")
RA0 = sp.Symbol("RA0")
RB0 = sp.Symbol("RB0")
S30 = sp.Symbol("S30")
IL10M = sp.Symbol("IL10M")
RA = sp.Symbol("RA")
RB = sp.Symbol("RB")
S3 = sp.Symbol("S3")

print("Before generation of final expressions")
RA_expr = RA0-RA-(rho_2+2*rho_3+rho_4+2*rho_5+rho_6+2*rho_7+2*rho_8+2*rho_9+2*rho_10)/rho_t*IL10M0
RB_expr = RB0-RB-(rho_4+rho_5+rho_6+rho_7+2*rho_8+2*rho_9+2*rho_10)/rho_t*IL10M0
S3_expr = S30-S3-(rho_6+rho_7+rho_9+2*rho_10)/rho_t*IL10M0
IL_expr = IL10M0-IL10M-(rho_1+rho_2+rho_3+rho_4+rho_5+rho_6+rho_7+rho_8+rho_9+rho_10)/rho_t*IL10M0

ILRA1RB1S31 = S30-S3-(rho_7+rho_9+2*rho_10)/rho_t*IL10M0
ILRA1RA2RB1S31 = S30-S3-(rho_6+rho_9+2*rho_10)/rho_t*IL10M0
ILRA1RA2RB1RB2S31 = S30-S3-(rho_6+rho_7+2*rho_10)/rho_t*IL10M0
ILRA1RA2RB1RB2S31S32 = (S30-S3-(rho_6+rho_7+rho_9)/rho_t*IL10M0)/2

with open("SS_expr_STAT3_simpl.pkl", "wb") as file:
    pickle.dump([str(RA_expr),str(RB_expr),str(S3_expr),str(IL_expr),str(ILRA1RB1S31),str(ILRA1RA2RB1S31),str(ILRA1RA2RB1RB2S31),str(ILRA1RA2RB1RB2S31S32)], file)
