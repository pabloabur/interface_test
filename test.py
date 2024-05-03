import snntorch as snn
from snntorch import spikegen, export_to_nir

import torch
import torch.nn as nn

from snntorch import spikeplot as splt
import matplotlib.pyplot as plt
import pprint

import nir

num_steps = 200

# layer parameters
num_inputs = 784
num_hidden = 1000
num_outputs = 10
beta = 0.9375
alpha = 0.8125

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

net = torch.nn.Sequential(fc1,
                          lif1,
                          fc2,
                          lif2
                          )

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

splt.traces(mem2_rec.squeeze(1), spk=spk2_rec.squeeze(1))
plt.show()

nir_graph = export_to_nir(net, spk_in)
nir.write("nir_graph.hdf5", nir_graph)
pprint.pp(nir_graph)
