from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

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

def visualise_connectivity_hardware(S, neurons_per_core, cores_per_pop):
    num_source_neu = len(S.i)

    # raw connections
    fig, ax = plt.subplots()
    # +1 for visualisation purposes
    plot_height = num_source_neu + 1
    plt.plot(np.zeros(num_source_neu),
         np.arange(num_source_neu),
         'ok', ms=10)
    plt.plot(np.ones(num_source_neu),
         np.arange(num_source_neu),
         'ok', ms=10)
    for i in np.arange(num_source_neu):
        plt.plot([0, 1], [i, i], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_visible(False)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, plot_height)

    cores = [neurons_per_core for _ in range(int(num_source_neu/neurons_per_core))]
    # TODO better way to calculate points where cores have their top height
    pre_cores = np.array_split(S.i, np.cumsum(cores))
    post_cores = np.array_split(S.j, np.cumsum(cores))

    # Drawing rectangles for hardware representation
    core_rect_height = plot_height / len(cores)
    # Manually adjusting height
    core_rect_height -= 0.1
    pop_rect_height = plot_height / (neurons_per_core * cores_per_pop)
    previous_rect = -0.5
    for _ in range(len(cores)):
        target_height = previous_rect+core_rect_height
        rect = matplotlib.patches.Rectangle((-0.02, previous_rect),
                                            0.1, core_rect_height,
                                            linewidth=1, edgecolor='r', facecolor='none')
        previous_rect += core_rect_height
        ax.add_patch(rect)

    plt.text(0.15, 0.15, '0')

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
visualise_connectivity_hardware(S, neurons_per_core, cores_per_pop)
plt.show()
