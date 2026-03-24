import sympy as sp
from collections import Counter

# Function that converts a PySB model into its linear framework graphs
def generate_graphs(model, graph_gen_species):
    # Get sympy symbols for all the model's species
    symbol_species = {}
    for i in range(0,len(model.species)):
        symbol_species["__s"+str(i)] = sp.Symbol("__s"+str(i)) # , positive=True
    
    # Get sympy symbol for species that generate the graphs of the system
    index_ggs = [i for i in range(0,len(model.species)) if model.species[i] in graph_gen_species]
    species_ggs = [sp.Symbol("__s"+str(i)) for i in index_ggs]
    
    # Convert the parameters
    param_symbols = {}
    for param_old in model.parameters:
        param_symbols[param_old] = sp.Symbol(str(param_old).split("'")[1])
    
    graphs_system = {}
    for index_graph in index_ggs:
        edges_graph = []
        symbol_graph = symbol_species["__s"+str(index_graph)]
        index_species_to_search = [index_graph]
        index_species_searched = []
        while len(index_species_to_search) > 0: # While still we have nodes/species that are needed to study
            index_searching = index_species_to_search[0]
            reactions_index = [i for i in range(0,len(model.reactions)) if index_searching in model.reactions[i]['reactants']]
            symbol_searching = symbol_species["__s"+str(index_searching)]
            for reaction in [model.reactions[i] for i in reactions_index]: # For reactions that have as a reactant the nodes/species we are searching
                reaction_rate = reaction["rate"]
                for param_old in param_symbols.keys(): # Convert the parameters of the reaction rates to sympy symbol
                    reaction_rate = reaction_rate.subs(param_old,param_symbols[param_old])
                reaction_product_list = [product for product in reaction["products"] if product not in [index for index in index_ggs if index != index_graph]] # Get the products that is not a monomer that generates another graph
                for reaction_product in reaction_product_list:
                    edges_graph.append((index_searching, reaction_rate/symbol_searching, reaction_product))
                    if reaction_product not in index_species_searched: # Don't search nodes/species already searched
                        if reaction_product not in index_species_to_search: # Don't search nodes/species just added to search list
                            if reaction_product not in index_ggs: # Don't search other monomers (graph generating species)
                                index_species_to_search.append(reaction_product)
            index_species_searched.append(index_species_to_search[0]) # Save searched index
            index_species_to_search = index_species_to_search[1:] # Take out nodes/species already searched
        graphs_system[symbol_graph] = edges_graph
        
    return graphs_system

# Function that fully adapts one graph from one of the graph generating sepcies so it can be used by rhos script
def adapt_graph_get_rhos(model, graphs_system, species_ggs_new, symbol_GGS, graph_gen_species):
    # Get sympy symbols for all the model's species
    symbol_species = {}
    for i in range(0,len(model.species)):
        symbol_species["__s"+str(i)] = sp.Symbol("__s"+str(i)) # , positive=True
    
    # Get sympy symbol for species that generate the graphs of the system
    index_ggs = [i for i in range(0,len(model.species)) if model.species[i] in graph_gen_species]
    species_ggs = [sp.Symbol("__s"+str(i)) for i in index_ggs]

    # Change the node numbers and edge 
    graph_str = []
    elements_graph = []
    for edge in graphs_system[symbol_GGS]:
        # Convert edges graph from sympy to string (Input for Rosa's code)
        reactant = edge[0]
        rate = edge[1]
        product = edge[2]
        for i in range(0,len(species_ggs)):
            rate = rate.subs(species_ggs[i],species_ggs_new[i])
        rate = str(rate).replace("1.0*","").replace("*","-")
        graph_str.append((reactant,rate,product))
    
        # Change node number to be continuous (0,1,2,3,4,...), get the new node numbers
        if edge[0] not in elements_graph:
            elements_graph.append(edge[0])
        elif edge[2] not in elements_graph:
            elements_graph.append(edge[2])
    elements_graph.sort()
    new_index_edges = {elements_graph[i]:i for i in range(0,len(elements_graph))}
    
    # Change node number to be continuous (0,1,2,3,4,...)
    graph_final = []
    for edge in graph_str:
        graph_final.append((new_index_edges[edge[0]]+1,edge[1],new_index_edges[edge[2]]+1))
    new_index_species = {new_index_edges[i]:model.species[i] for i in new_index_edges.keys()}

    # Change the parameter names so there are no weights that are the same
    weights_graph = [edge[1] for edge in graph_final]
    counts = Counter(weights_graph)
    repeated_weights = {item:0 for item, count in counts.items() if count > 1}
    dict_repeated_weights_names = {}
    for weight in repeated_weights.keys():
        for i in range(0,len(graph_final)):
            if weight == graph_final[i][1]:
                repeated_weights[weight] = repeated_weights[weight] + 1
                if len(graph_final[i][1].split("-")) == 1:
                    dict_repeated_weights_names[graph_final[i][1]+str(repeated_weights[weight])] = graph_final[i][1]
                    graph_final[i] = (graph_final[i][0], graph_final[i][1]+str(repeated_weights[weight]),graph_final[i][2])
                else:
                    dict_repeated_weights_names[graph_final[i][1].split("-")[0]+str(repeated_weights[weight])] = graph_final[i][1].split("-")[0]
                    graph_final[i] = (graph_final[i][0], graph_final[i][1].split("-")[0]+str(repeated_weights[weight])+"-"+graph_final[i][1].split("-")[1],graph_final[i][2])
                    
    return graph_final,new_index_edges,new_index_species,dict_repeated_weights_names
    
