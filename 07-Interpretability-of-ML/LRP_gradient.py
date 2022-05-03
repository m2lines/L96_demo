import numpy as np
import matplotlib.pyplot as plt
import math


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from torch import nn, optim
from torch_lr_finder import LRFinder
import torch.nn.functional as F

from sklearn.metrics import r2_score

np.random.seed(14)  # For reproducibility
torch.manual_seed(14)  # For reproducibility


from L96_model_XYtend import (
    L96,
    L96_eq1_xdot,
    integrate_L96_2t,
)  # L96_model_XYtend Adds the option to ouptput the subgrid tendencies (effect of Y on X)
from L96_model_XYtend import EulerFwd, RK2, RK4

## get the weight and bias of the NN
def get_weight(modelname):
    Ws = []
    Bs = []
    for (i, name) in enumerate(model.keys()):
        if i % 2 == 0:
            Ws.append(np.array(model[name]))
        else:
            Bs.append(np.array(model[name]))
    return Ws, Bs  # weights and biases


# forward pass to calculate the output of each layer
def forward_pass(data, Ws, Bs):
    L = len(Ws)
    forward = [data] + [None] * L

    for l in range(L - 1):
        forward[l + 1] = np.maximum(0, Ws[l].dot(forward[l])) + Bs[l]  # ativation ReLu

    ## for last layer that does not have activation function

    forward[L] = Ws[L - 1].dot(forward[L - 1]) + Bs[L - 1]  # linear last layer
    return forward


def rho(w, l):
    w_intermediate = w + [0.0, 0.0, 0.0, 0.0, 0.0][l] * np.maximum(0, w)
    return w_intermediate + gamma * np.maximum(0, w_intermediate)


def incr(z, l):
    return z + [0.0, 0.0, 0.0, 0.0, 0.0][l] * (z**2).mean() ** 0.5 + 1e-9


## backward pass to compute the LRP of each layer. Same rule applied to the first layer (input layer)
def onelayer_LRP(W, B, forward, nz, zz):
    mask = np.zeros((nz))
    mask[zz] = 1
    L = len(W)
    R = [None] * L + [forward[L] * mask]  # start from last layer Relevance

    for l in range(0, L)[::-1]:
        w = rho(W[l], l)
        b = rho(B[l], l)
        z = incr(w.dot(forward[l]) + b + epsilon, l)  # step 1 - forward pass
        s = np.array(R[l + 1]) / np.array(z)  # step 2 - element-wise division
        c = w.T.dot(s)  # step 3 - backward pass
        R[l] = forward[l] * c  # step 4 - element-wise product

    return R


def LRP_alllayer(data, model):
    """inputs:
        data: for single sample, with the right asix, the shape is (nz,naxis)
        model: dictionary of weights and biases
    output:
        LRP, shape: (nx,L+1) that each of the column consist of L+1 array
        Relevance of fisrt layer's pixels"""
    nx = data.shape[0]
    ## step 1: get the wieghts
    Ws, Bs = get_weight(model)

    ## step 2: call the forward pass to get the intermediate layers output
    inter_layer = forward_pass(data, Ws, Bs)

    ## loop over all z and get the LRP of each layer
    R_all = [None] * nx
    relevance = np.zeros((nx, nx))
    for xx in range(nx):
        R_all[xx] = onelayer_LRP(Ws, Bs, inter_layer, nx, xx)
        relevance[xx, :] = R_all[xx][0]

    return np.stack(R_all), relevance


######## main code
time_steps = 20000
Forcing, dt, T = 18, 0.01, 0.01 * time_steps

# Create a "synthetic world" with K=8 and J=32
K = 8
J = 32
W = L96(K, J, F=Forcing)
# Get training data for the neural network.

# - Run the true state and output subgrid tendencies (the effect of Y on X is xytrue):
Xtrue, _, _, xytrue = W.run(dt, T, store=True)


# Specify a path
PATH = "../04Subgrid-parametrization-pytorch/networks/network_3_layers_100_epoches.pth"
# Load
model = torch.load(PATH)
model.keys()

for name in model.keys():
    print(name)


epsilon = 0.0  # filtering small values
gamma = 0.0  # give more weights to positive values


R_many = []
for case in range(200):
    inputs = np.copy(Xtrue[case, :])

    _, Rs = LRP_alllayer(inputs, model)
    R_many.append(Rs)

Rstack = np.stack(R_many)


fig, ax = plt.subplots(1, 1)
vect = np.arange(-5, 5.1, 0.5)
pl = ax.contourf(
    np.arange(1, 9),
    np.arange(1, 9),
    np.mean(Rstack, 0),
    vect,
    cmap=plt.get_cmap("seismic"),
)
ax.set_ylabel("Output layer", fontsize=12)
ax.set_xlabel("Input Layer relevance", fontsize=12)
ax.tick_params(axis="both", labelsize=12)
fig.colorbar(pl)
fig.savefig("LRP-L96.jpeg", bbox_inches="tight", dpi=150)
