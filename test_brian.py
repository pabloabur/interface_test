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

def create_lookup_table(S, neurons_per_core, cores_per_pop):
    num_source_neu = len(S.i)

    cores = [neurons_per_core for _ in range(int(num_source_neu/neurons_per_core))]
    pre_cores = np.array_split(S.i, np.cumsum(cores))
    post_cores = np.array_split(S.j, np.cumsum(cores))

    # TODO populate it
    df = pd.DataFrame({
        "source_index": S.i,
        "target_index": S.j,
        "core_index": ,
        "population_index": ,
        "neural_type": ,
        })

N = 10
#con_prob = 0.2
con_prob = 0.5
#con_prob = 1.0
G = NeuronGroup(N, 'v:1')
S = Synapses(G, G)
S.connect(condition='i!=j', p=con_prob)

# Hardware mapping
neurons_per_core = 2
cores_per_pop = 2

visualise_connectivity(S)
create_lookup_table(S, neurons_per_core, cores_per_pop)
plt.show()
