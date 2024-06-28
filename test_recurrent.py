import snntorch as snn
from snntorch import spikegen#, export_to_nir

import torch
import torch.nn as nn

from snntorch import spikeplot as splt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pprint

#import nir

num_steps = 200

# layer parameters
num_inputs = 285
Ne, Ni = 3471, 613
num_hidden = Ne+Ni
beta = 0.9375
alpha = 0.8125
w_ex = 0.125
w_in = 5.5*w_ex


def traces(data, spk=None, dim=(3, 3), spk_height=5, titles=None):
    """Plot an array of neuron traces (e.g., membrane potential or synaptic
    current).
    Optionally apply spikes to ride on the traces.
    `traces` was originally written by Friedemann Zenke.

    Example::

        import snntorch.spikeplot as splt

        #  mem_rec contains the traces of 9 neuron membrane potentials across
        100 time steps in duration
        print(mem_rec.size())
        >>> torch.Size([100, 9])

        #  Plot
        traces(mem_rec, dim=(3,3))


    :param data: Data tensor for neuron traces across time steps of shape
        [num_steps x num_neurons]
    :type data: torch.Tensor

    :param spk: Data tensor for neuron traces across time steps of shape
        [num_steps x num_neurons], defaults to ``None``
    :type spk: torch.Tensor, optional

    :param dim: Dimensions of figure, defaults to ``(3, 3)``
    :type dim: tuple, optional

    :param spk_height: height of spike to plot, defaults to ``5``
    :type spk_height: float, optional

    :param titles: Adds subplot titles, defaults to ``None``
    :type titles: list of strings, optional

    """

    gs = GridSpec(*dim)

    if spk is not None:
        data = (data + spk_height * spk).detach().cpu().numpy()

    else:
        data = data.detach().cpu().numpy()

    for i in range(np.prod(dim)):
        if i == 0:
            a0 = ax = plt.subplot(gs[i])
            if titles is not None:
                ax.set_title(titles[i])

        else:
            ax = plt.subplot(gs[i], sharey=a0)
            if titles is not None and i < len(titles):
                ax.set_title(titles[i])

        ax.plot(data[:, i])


# initialize layers
# TODO define dt
fc1 = nn.Linear(num_inputs, num_hidden, bias=False)
lif1 = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True, init_hidden=True,
                     linear_features=num_hidden, learn_recurrent=False,
                     reset_mechanism="zero", threshold=20,
                     output=True)
lif1.recurrent = nn.Linear(lif1.linear_features, lif1.linear_features, bias=False)
lif1.reset_mem()

# Connectivity probability and weights
seed = 42
rng = np.random.default_rng(seed)
pre_size, post_size = num_hidden, num_hidden
prob_conn = 0.1
conn_mat = rng.choice([0, 1],
                      size=(pre_size, post_size),
                      p=[1-prob_conn, prob_conn])
np.fill_diagonal(conn_mat, 0)
sources, targets = conn_mat.nonzero()
conn_mat = torch.tensor(conn_mat, dtype=torch.float)

e_sources = sources[sources<Ne]
we_init = torch.normal(w_ex, w_ex/10, size=(1, len(e_sources)))
conn_mat[e_sources, targets[:len(e_sources)]] = we_init
i_sources = sources[len(e_sources):]
wi_init = -torch.normal(w_in, w_in/10, size=(1, len(i_sources)))
conn_mat[i_sources, targets[len(e_sources):]] = wi_init
lif1.recurrent.weight.data = conn_mat

# record outputs
spk2_rec = []
syn2_rec = []
mem2_rec = []

# Input
dt = 1e-3 # each timestep is 1ms
input_rate = 6 # Hz
spike_gen_prob = input_rate * dt
spk_in = spikegen.rate(torch.Tensor([spike_gen_prob for _ in range(num_inputs)]),
                       num_steps=num_steps)

pre_size, post_size = num_inputs, num_hidden
prob_conn = 0.25
input_conn_mat = rng.choice([0, 1],
                            size=(pre_size, post_size),
                            p=[1-prob_conn, prob_conn])
sources, targets = input_conn_mat.nonzero()
input_conn_mat = torch.tensor(input_conn_mat, dtype=torch.float)

winp_init = torch.normal(w_ex, w_ex/10, size=(1, len(sources)))
input_conn_mat[sources, targets] = winp_init
fc1.weight.data = input_conn_mat.transpose(0, 1)

net = torch.nn.Sequential(fc1, lif1)

# network simulation
for step in range(num_steps):
    spk1, syn1, mem1 = net(spk_in[step])

    spk2_rec.append(spk1)
    syn2_rec.append(syn1)
    mem2_rec.append(mem1)

# convert lists to tensors
spk2_rec = torch.stack(spk2_rec)
syn2_rec = torch.stack(syn2_rec)
mem2_rec = torch.stack(mem2_rec)

#traces(mem2_rec.squeeze(1), spk=spk2_rec.squeeze(1))
plt.figure()
plt.title("Vm")
plt.plot(mem2_rec[:, 2].detach())
plt.show()

plt.figure()
plt.title("Syn")
plt.plot(syn2_rec[:, 2].detach())
plt.show()

plt.figure()
plt.title("Rec weights")
plt.imshow(lif1.recurrent.weight.data)
plt.colorbar()
plt.show()

plt.figure()
plt.title("Input weights")
plt.imshow(fc1.weight.data.transpose(0, 1))
plt.colorbar()
plt.show()

ids, times = spk2_rec.detach().numpy().transpose().nonzero()
plt.figure()
plt.title("Neuron spikes")
plt.plot(times, ids, '.')
plt.show()

ids, times = spk_in.detach().numpy().transpose().nonzero()
plt.figure()
plt.title("Input spikes")
plt.plot(times, ids, '.')
plt.show()
#nir_graph = export_to_nir(net, spk_in)
#nir.write("nir_graph.hdf5", nir_graph)
#pprint.pp(nir_graph)
