from brian2 import NeuronGroup, Synapses, Network, collect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def preprocess_network(network):
    """ Preprocesses a Network object from Brian2. 

    Arguments
    ---------
    network : brian2.Network
        A Network object containing NeuronGroup and Synapses.

    Returns
    -------
    global_neuron_id : np.array
        A numpy array of global neuron IDs.
    global_synapse_ids : np.array
        A 2D numpy array of global synapse IDs.
    current_table : np.array
        A 1D numpy array of current values.
    voltage_table : np.array
        A 1D numpy array of voltage values.
    trace_table : np.array
        A 1D numpy array of trace values.

    Notes
    -----
    The order of neuron groups depends on the order of the objects in Network.
    The order is arbitrary, so the indices can change for different simulations.
    """
    neurons = [x for x in net.objects if isinstance(x, NeuronGroup)]
    num_neurons = sum([x.N for x in neurons])
    synapses = [x for x in net.objects if isinstance(x, Synapses)]

    current_table = np.zeros(num_neurons)
    voltage_table = np.zeros(num_neurons)
    trace_table = np.zeros(num_neurons)

    global_neuron_id = np.array([])
    # In order of appearance in list, regardless of NeuronGroup initialisation
    for neu in neurons:
        temp_length = len(global_neuron_id)
        temp_slice = temp_length + neu.i
        global_neuron_id = np.concatenate((global_neuron_id, temp_slice))
        current_table[temp_slice] = neu.get_states(['I'])['I']
        voltage_table[temp_slice] = neu.get_states(['v'])['v']
        trace_table[temp_slice] = neu.get_states(['x'])['x']

    global_synapse_ids = np.array([]).reshape(0, 2)
    for syn in synapses:
        source_group = syn.source
        target_group = syn.target
        offset_source = 0
        offset_target = 0
        for neu in neurons:
            if neu == source_group:
                break
            offset_source += neu.N
        for neu in neurons:
            if neu == target_group:
                break
            offset_target += neu.N

        temp_synapse_ids = np.hstack((
            global_neuron_id[syn.i + offset_source][:, np.newaxis],
            global_neuron_id[syn.j + offset_target][:, np.newaxis]))
        global_synapse_ids = np.vstack((global_synapse_ids, temp_synapse_ids))

    return (global_neuron_id, global_synapse_ids, current_table,
            voltage_table, trace_table)

def plot_matrix(state_variables, num_cores, num_subcores):
    """ Function used to visualise state variables stored in memory """
    plt.figure()
    plt.imshow(state_variables, aspect='auto', origin='lower', cmap='Greys')
    core_height = state_variables.shape[0] / num_cores
    core_width = state_variables.shape[1]
    subcore_height = core_height / num_subcores
    subcore_width = core_width

    for i in range(num_cores):
        plt.gca().add_patch(plt.Rectangle((-.48, i*core_height), core_width, core_height, fill=False, edgecolor='b', linewidth=2))

    for i in range(num_cores*num_subcores):
        plt.gca().add_patch(plt.Rectangle((-.48, i*subcore_height), subcore_width,
                                          subcore_height, fill=False, edgecolor='r',
                                          linewidth=2, linestyle=':'))
    plt.show()

def get_core(state_variables, core_num, num_subcores, num_state_vars):
    """ Get state variables from a core

    Arguments
    ---------
    state_variables : 2D np.array
        State variables stored in memory
    core_num : int
        The core number
    num_subcores : int
        Number of subcores
    num_state_vars : int
        Number of state variables

    Returns
    -------
    2D np.array
        State variables from a core

    Notes
    -----
    State variables are stored contiguously in rows of the matrix. Each row
    contains a number of neurons as deterimed by the number of subcores.
    """
    core_rows = num_subcores*num_state_vars
    bottom_id = state_variables.shape[0]
    return state_variables[bottom_id - (core_num + 1)*core_rows : bottom_id - core_num*core_rows, :]

def get_vars(state_variables, index, num_state_vars):
    """ Get state variables from a neuron in a core.

    Arguments
    ---------
    state_variables : 2D np.array
        State variables in a core
    index : int
        The index of the neuron in core
    num_state_vars : int
        Number of state variables

    Returns
    -------
    1D np.array
        State variables from a neuron
    """
    max_hight = state_variables.shape[0]
    num_cols = state_variables.shape[1]
    col = index % num_cols
    row = max_hight - (index // num_cols)*num_state_vars
    return state_variables[row-num_state_vars:row, col]

def create_lookup_table(neurons, synapses, voltages, currents, traces,
                        neurons_per_core, num_cores, num_subcores, num_state_vars):
    """ Create lookup table for state variables

    Arguments
    ---------
    neurons_per_core : int
        Number of neurons per core
    neurons : 1D np.array
        List of indices of NeuronGroup objects
    synapses : 2D np.array
        Each row contains source and target neuronal indices
    voltages : 1D np.array
        Voltages stored in memory
    currents : 1D np.array
        Currents stored in memory
    traces : 1D np.array
        Traces stored in memory
    num_subcores : int
        Number of subcores
    num_state_vars : int
        Number of state variables
    num_cores : int
        Number of cores

    Returns
    -------
    state_variables : 2D np.array
        State variables organised in memory as cores
    pointer_table : 2D np.array
        Addresses for each connection. Each row is one sample. Columns are
        organised from left to right as: source core, source intracore index,
        target core, target intracore index.
    """
    num_neurons = len(neurons)
    core_index = [x for x in range(num_cores)]
    # Some cores may not be fully occupied, so we fix sizes here
    mismatch_size = abs(num_neurons - num_cores*neurons_per_core)
    if mismatch_size:
        # extend numpy array with NaN instead of list as below
        voltages = np.append(voltages, [np.nan for _ in range(mismatch_size)])
        currents = np.append(currents, [np.nan for _ in range(mismatch_size)])
        traces = np.append(traces, [np.nan for _ in range(mismatch_size)])
    intracore_index = [x for _ in range(num_cores) for x in range(neurons_per_core)]

    # Each subcore in a core memory matrix has neurons_per_subcore. 8 by default
    neurons_per_subcore = int(np.ceil(neurons_per_core / num_subcores))
    # Considering 3 state variables, each with 8-bits. 
    matrix_num_rows = num_state_vars * num_cores * num_subcores
    state_variables = np.empty((matrix_num_rows, neurons_per_subcore))

    split_indices = [x for x in range(neurons_per_subcore, len(voltages),
                                      neurons_per_subcore)]
    split_voltages = np.split(voltages, split_indices)
    split_currents = np.split(currents, split_indices)
    split_traces = np.split(traces, split_indices)
    # Memory is populated from bottom to top
    temp_index = -1
    for (v, i, x) in zip(split_voltages, split_currents, split_traces):
        # if v is none, others will also be
        if not any(v):
            break
        state_variables[temp_index, :] = v
        state_variables[temp_index-1, :] = i
        state_variables[temp_index-2, :] = x
        temp_index -= 3

    pointer_table = np.zeros((synapses.shape[0], 4))
    for i, syn in enumerate(synapses):
        source_core = int(syn[0] / neurons_per_core)
        intracore_source = syn[0] % neurons_per_core
        target_core = int(syn[1] / neurons_per_core)
        intracore_target = syn[1] % neurons_per_core
        pointer_table[i, :] = [source_core, intracore_source, target_core, intracore_target]

    return state_variables, pointer_table

net_type = 2
con_prob = 0.05
#con_prob = 0.5
#con_prob = 1.0
if net_type == 1:
    G1 = NeuronGroup(10,
                     '''v:1
                        I:1
                        x:1''',
                     name="population_1")
    G1.v = 'int(255*rand())'
    G1.I = 'int(255*rand())'
    G1.x = 'int(255*rand())'
    G2 = NeuronGroup(11,
                     '''v:1
                        I:1
                        x:1''',
                     name="population_2")
    G2.v = 'int(255*rand())'
    G2.I = 'int(255*rand())'
    G2.x = 'int(255*rand())'
    G3 = NeuronGroup(9,
                     '''v:1
                        I:1
                        x:1''',
                     name="population_3")
    G3.v = 'int(255*rand())'
    G3.I = 'int(255*rand())'
    G3.x = 'int(255*rand())'
    S = Synapses(G1, G2)
    S.connect(p=con_prob)
    S1 = Synapses(G1, G3)
    S1.connect(p=con_prob)
    S2 = Synapses(G2, G3)
    S2.connect(p=con_prob)
    Sr = Synapses(G1, G1)
    Sr.connect(p=con_prob)
elif net_type == 2:
    G1 = NeuronGroup(1024,
                     '''v:1
                        I:1
                        x:1''',
                     name="population_1")
    G1.v = 'int(255*rand())'
    G1.I = 'int(255*rand())'
    G1.x = 'int(255*rand())'
    G2 = NeuronGroup(1024,
                     '''v:1
                        I:1
                        x:1''',
                     name="population_2")
    G2.v = 'int(255*rand())'
    G2.I = 'int(255*rand())'
    G2.x = 'int(255*rand())'
    S = Synapses(G1, G2)
    S.connect(p=con_prob)

net = Network(collect())

neu_id, syn_id, currents, voltages, traces = preprocess_network(net)

# Hardware mapping
if net_type == 1:
    neurons_per_core = 16
elif net_type == 2:
    neurons_per_core = 512
num_subcores = 8
num_cores = int(np.ceil(len(neu_id)/neurons_per_core))
num_state_vars = 3

state_vars, pointer_table = create_lookup_table(neu_id, syn_id,
                                                voltages, currents, traces,
                                                neurons_per_core, num_cores,
                                                num_subcores, num_state_vars)
plot_matrix(state_vars, num_cores, num_subcores)
plt.show()

# How to access memory
target_core = 0
target_intracore = 1
neu_vars = get_vars(get_core(state_vars, target_core, num_subcores, num_state_vars),
                    target_intracore, num_state_vars)
print(neu_vars)
