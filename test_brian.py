from brian2 import NeuronGroup, Synapses, Network, collect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=10)
    plt.plot(np.ones(Nt), np.arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')

def create_lookup_table(neurons_per_core, network, recurrent_index):
    num_neurons = sum([x.N for x in network.objects if isinstance(x, NeuronGroup)])
    num_cores = int(np.ceil(num_neurons/neurons_per_core))
    core_index = [x for x in range(num_cores)]
    # Some cores may not be fully occupied, so we fix sizes here
    neuron_index = [x for x in range(num_neurons)]
    mismatch_size = abs(num_neurons - num_cores*neurons_per_core)
    if mismatch_size:
        neuron_index.extend([np.nan for _ in range(mismatch_size)])
    intracore_index = [x for _ in range(num_cores) for x in range(neurons_per_core)]

    # Each subcore in a core memory matrix has neurons_per_subcore. 8 by default
    num_subcores = 8
    neurons_per_subcore = int(np.ceil(neurons_per_core / num_subcores))
    # Considering 3 state varibles, each with 8-bits. 
    matrix_num_rows = 3 * num_subcores
    state_variables = np.empty((num_cores*matrix_num_rows, neurons_per_subcore),
                               dtype=np.int8)

    def map_synapses(syn):
        # TODO
        # TODO indices relative to groups and I don't know total number of neurons
        # 	maps neurons to matrix and manage core/neu map?
        #	maybe anchor target neuron group and build map interactively
        # [syn.target==neurons[0] for syn in synapses]?
        pass

    def set_states(states, target_neurons):
        """
        Parameters
        ----------
        states : ndarray
            Matrix with state variables.
        target_neurons : ndarray
        -------
        """
        pass
    neurons = [x for x in network.objects if isinstance(x, NeuronGroup)]
    index_pointer = 0
    for neu in neurons:
        states = neu.get_states(['v', 'x', 'I'])
        for state in states:
            pass

    import pdb; pdb.set_trace()
    #np.array_split(neurons[0].get_states(['v', 'x', 'I'])['v'], np.cumsum([neurons_per_core for _ in range(num_blocks)]))

    df_neurons = pd.DataFrame({
        "neuron": intracore_index,
        "core": core_index,
        })

    # TODO get neuron and synapse groups from net. syn.N from i or j. need to map groups
    # TODO map neurons to memory
    # TODO map synapses
    synapses = [x for x in network.objects if isinstance(x, Synapses)]
    # TODO two different populations targetting the same population should not create new entries in matrices created here
    for syn in synapses:
        if recurrent_index:
            # TODO recurrent connectivity pattern
            pass

    source_id = np.array(S.i).tolist()
    target_id = np.array(S.j).tolist()
    source_core = df_neurons.loc[source_id]['core_index'].to_numpy()
    source_internal_index = df_neurons.loc[source_id]['intracore_index'].to_numpy()
    target_core = df_neurons.loc[target_id]['core_index'].to_numpy()
    target_internal_index = df_neurons.loc[target_id]['intracore_index'].to_numpy()

    df_synapses = pd.DataFrame({
        "source_id": source_id,
        "source_core": source_core,
        "source_internal_index": source_internal_index,
        "target_id": target_id,
        "target_core": target_core,
        "target_internal_index": target_internal_index,
        })

    return df_neurons, df_synapses

num_neurons = 10
#con_prob = 0.2
con_prob = 0.5
#con_prob = 1.0
G1 = NeuronGroup(num_neurons,
                 '''v:1
                    I:1
                    x:1''',
                 name="population_1")
G2 = NeuronGroup(num_neurons,
                 '''v:1
                    I:1
                    x:1''',
                 name="population_2")
S = Synapses(G1, G2)
S.connect(p=con_prob)
Sr = Synapses(G1, G1)
Sr.connect(p=con_prob)

# TODO another model:1k -> 1k, with 1-2 (if recurrent), 1-1024, 1-1 (to other pop), 1024-1 (to other pop)

net = Network(collect())

# Hardware mapping
neurons_per_core = 16

visualise_connectivity(S)
df_neurons, df_synapses = create_lookup_table(neurons_per_core, net, [1])
plt.show()
