from brian2 import NeuronGroup, Synapses
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

def create_lookup_table(num_neurons, neurons_per_core, cores_per_pop):
    neurons_per_pop = neurons_per_core * cores_per_pop
    population_slices = int(np.ceil(num_neurons/neurons_per_pop))
    core_slices = int(neurons_per_pop/neurons_per_core)

    population_index = [x for x in range(population_slices) for _ in range(neurons_per_pop)]
    population_index = np.array(population_index)

    core_index = [x for x in range(core_slices) for _ in range(neurons_per_core)]
    core_index = np.tile(core_index, population_slices)

    # Some populations may not be fully occupied, so we fix sizes here
    neuron_index = [x for x in range(num_neurons)]
    mismatch_size = abs(num_neurons-len(core_index))
    if mismatch_size:
        neuron_index.extend([np.nan for _ in range(mismatch_size)])

    internal_index = [x for _ in range(core_slices) for x in range(neurons_per_core)]
    internal_index = np.tile(internal_index, population_slices)

    df_neurons = pd.DataFrame({
        "neuron_index": neuron_index,
        "internal_index": internal_index,
        "core_index": core_index,
        "population_index": population_index,
        })

    source_id = np.array(S.i).tolist()
    target_id = np.array(S.j).tolist()
    source_population = df_neurons.loc[source_id]['population_index'].to_numpy()
    source_core = df_neurons.loc[source_id]['core_index'].to_numpy()
    source_internal_index = df_neurons.loc[source_id]['internal_index'].to_numpy()
    target_population = df_neurons.loc[target_id]['population_index'].to_numpy()
    target_core = df_neurons.loc[target_id]['core_index'].to_numpy()
    target_internal_index = df_neurons.loc[target_id]['internal_index'].to_numpy()

    import pdb;pdb.set_trace()

    df_synapses = pd.DataFrame({
        "source_id": source_id,
        "source_population": source_population,
        "source_core": source_core,
        "source_internal_index": source_internal_index,
        "target_id": target_id,
        "target_population": target_population,
        "target_core": target_core,
        "target_internal_index": target_internal_index,
        })

num_neurons = 10
#con_prob = 0.2
con_prob = 0.5
#con_prob = 1.0
G = NeuronGroup(num_neurons, 'v:1')
S = Synapses(G, G)
S.connect(condition='i!=j', p=con_prob)

# Hardware mapping
neurons_per_core = 2
cores_per_pop = 2

visualise_connectivity(S)
create_lookup_table(num_neurons, neurons_per_core, cores_per_pop)
plt.show()
