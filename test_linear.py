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
num_inputs = 784
num_hidden = 1000
num_outputs = 10
beta = 0.9375
alpha = 0.8125


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
# TODO reset need to be correct
# TODO define populations (as in layers) or use single with metadata
# TODO identify static or plastic weights somehow
# TODO why do I have "input" and "output" in my graph?
fc1 = nn.Linear(num_inputs, num_hidden)
lif1 = snn.Synaptic(alpha=alpha, beta=beta, init_hidden=True)
fc2 = nn.Linear(num_hidden, num_outputs)
lif2 = snn.Synaptic(alpha=alpha, beta=beta, init_hidden=True, output=True)

# Initialize hidden states
mem1, syn1 = lif1.init_synaptic()
mem2, syn2 = lif2.init_synaptic()

# record outputs
mem2_rec = []
spk2_rec = []

spk_in = spikegen.rate_conv(torch.rand((200, 784))).unsqueeze(1)

net = torch.nn.Sequential(fc1, lif1, fc2, lif2)

# network simulation
spk_in = torch.rand(num_steps, 784)
for step in range(num_steps):
    # TODO define weights (from input and neurons): class has a weight attr
    spk2, syn2, mem2 = net(spk_in[step])

    mem2_rec.append(mem2)
    spk2_rec.append(spk2)

# convert lists to tensors
mem2_rec = torch.stack(mem2_rec)
spk2_rec = torch.stack(spk2_rec)

traces(mem2_rec.squeeze(1), spk=spk2_rec.squeeze(1))
plt.show()

#nir_graph = export_to_nir(net, spk_in)
#nir.write("nir_graph.hdf5", nir_graph)
#pprint.pp(nir_graph)
