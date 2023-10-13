import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

import torch
from torch.autograd import Variable
import torch.utils.data as Data
from torch import nn, optim

from L96 import L96, L96_eq1_xdot, integrate_L96_2t, EulerFwd, RK2, RK4


class GCM_no_param:
    def __init__(self, F, time_stepping=RK4, a=1):
        self.F = F
        self.time_stepping = time_stepping
        self.a = a

    def rhs(self, X):
        return L96_eq1_xdot(X, self.F, a=self.a)

    def __call__(self, X0, dt, nt):
        # X0 - initial conditions, dt - time increment, nt - number of forward steps to take
        time, hist, X = (
            dt * np.arange(nt + 1),
            np.zeros((nt + 1, len(X0))) * np.nan,
            X0.copy(),
        )
        hist[0] = X

        for n in range(nt):
            X = self.time_stepping(self.rhs, dt, X)
            hist[n + 1], time[n + 1] = X, dt * (n + 1)
        return hist, time


# New GCM: no parameterization and global network for tendency
class GCM_discrete:
    def __init__(self, net):
        self.net = net

    def next_step(self, X):
        if torch.is_tensor(X):
            return self.net(X)
        else:
            return self.net(torch.from_numpy(X)).data.numpy()

    def __call__(self, X0, nt):
        # X0 - initial conditions
        # nt - number of forward steps to take

        # We add extra axes here to accompany CNN
        if len(X0.shape) == 2:
            X = X0[np.newaxis, :]
        elif len(X0.shape) == 1:
            X = X0[np.newaxis, np.newaxis, :]
        else:
            X = X0.copy()

        # Raise into pytorch
        X = torch.from_numpy(X)

        # Step forward and write down the states
        hist = np.zeros((nt + 1, len(X0))) * np.nan
        hist[0] = X.squeeze()
        for n in range(nt):
            X = X + self.next_step(X).detach().numpy()
            hist[n + 1] = X.squeeze()
        return hist


# New GCM: no parameterization and local stencil for tendency
class GCM_local_discrete:
    def __init__(self, net, get_features):
        self.net = net
        self.get_features = get_features

    def next_step(self, X):
        # X need to have shape [time, space]
        n = len(self.net.filter_loc)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_rolled = np.zeros((X.shape[1], n))
        for iroll in range(n):
            X_rolled[:, iroll] = np.roll(X, self.net.filter_loc[iroll], axis=1)

        # Obtain polynomials
        if self.get_features == None:
            X_torch = torch.from_numpy(X_rolled)
        else:
            X_torch = torch.from_numpy(self.get_features(X_rolled))

        return self.net(X_torch).data.numpy().T

    def __call__(self, X0, nt):
        # X0 - initial conditions
        # nt - number of forward steps to take
        hist, X = np.zeros((nt + 1, len(X0))) * np.nan, X0.copy()
        hist[0, :] = X

        for n in range(nt):
            X = X + self.next_step(X)
            hist[n + 1, :] = X
        return hist


class Net_ANN(nn.Module):
    def __init__(self, neuron_dims, filter_loc=np.array([-2, -1, 0, 1, 2])):
        super(Net_ANN, self).__init__()
        self.filter_loc = filter_loc
        self.layers = list()
        self.neuron_dims = neuron_dims
        for i in range(len(self.neuron_dims) - 1):
            self.layers.append(nn.Linear(self.neuron_dims[i], self.neuron_dims[i + 1]))

        # Warp the list of layers
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for i in range(len(self.neuron_dims) - 2):
            x = torch.tanh(self.layers[i](x))
        x = self.layers[-1](x)
        return x


class Net_CNN(nn.Module):
    def __init__(
        self,
        inout_size=[100, 100],
        kernel_size=5,
        hidden_neurons=[20, 20],
        strides=[1, 1],
        channels=[1, 1],
    ):
        super(Net_CNN, self).__init__()
        # Class of a very simple 1-dimensional convolutional neural network. This network contains
        # 2 convolutional layers of 1 channel each, and 2 fully-connected layers after the convolutional
        # layers.

        # Input parameters:
        #
        # inout_size:     A 2-element list, the first element contains the input size;
        #                 the second element contains the output size.
        #
        # kernel_size:    The size of the convolutional kernel. Same across all conv layers
        #
        # hidden_neurons: List of hidden neurons of the fully-connected layers after the conv layers

        # Set padding to half the size of the kernel, and padding mode to periodic (circular)
        padding = kernel_size // 2
        padding_mode = "circular"

        # Set dilation to 1, i.e., no dilation
        dilation = 1

        self.conv1 = torch.nn.Conv1d(
            1,
            channels[0],
            kernel_size,
            stride=strides[0],
            padding=padding,
            padding_mode=padding_mode,
        )

        self.conv2 = torch.nn.Conv1d(
            channels[0],
            channels[1],
            kernel_size,
            stride=strides[1],
            padding=padding,
            padding_mode=padding_mode,
        )

        # Define a function to calculate the dimensionality of the input to the fully-connected layer
        # (see definition of L_out in https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
        output_size = lambda L_in, padding, dilation, kernel_size, stride: int(
            (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )
        output_size1 = output_size(
            inout_size[0], padding, dilation, kernel_size, strides[0]
        )
        output_size2 = output_size(
            output_size1, padding, dilation, kernel_size, strides[1]
        )

        self.output_size2 = output_size2
        self.channels = channels

        self.linear1 = nn.Linear(output_size2 * channels[1], hidden_neurons[0])
        self.linear2 = nn.Linear(hidden_neurons[0], hidden_neurons[1])
        self.linear3 = nn.Linear(hidden_neurons[1], inout_size[1])

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 1, self.channels[1] * self.output_size2)
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x


def burgers_tendency_cd(u, dx, nu):
    """
    Calculate the tendency of u in the Burger's equation under spatial central difference:
    du/dt[k] = - u[k]*(u[k+1] - u[k-1])/2/dx + nu*(u[k+1] - 2*u[k] + u[k-1])/dx**2
    Args:
        u : Array of u at the current time step
        nu : Diffusivity
    Returns:
        dudt : Array of u time tendencies
    """
    K = len(u)
    dudt = np.zeros(K)
    dudt = (
        -u * (np.roll(u, -1) - np.roll(u, 1)) / 2 / dx
        + nu * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
    )

    return dudt


def integrate_burgers(u0, si, nt, dx, nu, method=RK4, t0=0, dt=0.001):
    """
    Integrates forward-in-time the Burger's equation using the integration "method".
    Returns the full history with nt+1 values starting with initial conditions, u[:,0]=u0, and ending
    with the final state, u[:,nt+1] at time t0+nt*dt.

    Args:
        u     : Values of u variables at the current time
        dt     : The time step
        nt     : Number of forwards steps
        method : The time-stepping method that returns u(n+1) given u(n)
        t0     : Initial time (defaults to 0)

    Returns:
        u[:,:], time[:] : the full history u[n,k] at times t[n]

    Example usage:
        u, t = integrate_burgers(5+5*np.random.rand(8), 18, 0.01, 500)
        plt.plot(t, u);
    """

    time, hist = t0 + np.zeros((nt + 1)), np.zeros((nt + 1, len(u0)))
    u = u0.copy()
    hist[0, :] = u
    if si < dt:
        dt, ns = si, 1
    else:
        ns = int(si / dt + 0.5)
        assert (
            abs(ns * dt - si) < 1e-14
        ), "si is not an integer multiple of dt: si=%f dt=%f ns=%i" % (si, dt, ns)

    for n in range(nt):
        for s in range(ns):
            u = method(burgers_tendency_cd, dt, u, dx, nu)
            hist[n + 1, :], time[n + 1] = u, t0 + si * (n + 1)
    return hist, time


# Class for convenience
class Burgers:
    """
    Class for 1D Burger's equation
    """

    u = "Current u state or initial conditions"
    dx = "Gird spacing"
    nu = "Diffusivity"
    dt = "Time step"
    u_init = "Initial condition of u"

    def __init__(self, K, nu, dx, t=0, dt=0.001, u_init=None, method=RK4):
        """Construct a two time-scale model with parameters:
        K  : Number of u values
        nu : Diffusivity
        dx : Spatial grid spacing
        t  : Initial time (default 0.)
        dt : Time step (default 0.001)
        """
        self.dx, self.nu, self.dt = dx, nu, dt
        self.t = t
        self.method = method
        if u_init is None:
            x = np.arange(0, K)
            self.u_init = np.exp(-(((x - x.mean()) / K * 10) ** 2))
            np.random.randn(K)
        self.u = self.u_init
        self.K = K

    def __repr__(self):
        return (
            "Burgers: "
            + "K="
            + str(self.K)
            + " dx="
            + str(self.dx)
            + " nu="
            + str(self.nu)
            + " dt="
            + str(self.dt)
        )

    def __str__(self):
        return self.__repr__() + "\n u=" + str(self.u) + "\n t=" + str(self.t)

    def print(self):
        print(self)

    def set_param(self, dt=None, dx=None, nu=None, t=0):
        """Set a model parameter, e.g. .set_param(dt=0.002)"""
        if dt is not None:
            self.dt = dt
        if nu is not None:
            self.nu = nu
        if dx is not None:
            self.dx = dx
        if t is not None:
            self.t = t
        return self

    def set_state(self, u, t=None):
        """Set initial conditions (or current state), e.g. .set_state(u)"""
        self.u = u
        if t is not None:
            self.t = t
        self.K = self.u.size  # For convenience
        return self

    def set_init(self, u_init):
        self.u_init = u_init.reshape(self.u_init.shape)
        self.u = self.u_init

    def run(self, si, T, store=False, method=RK4):
        """Run model for a total time of T
        If store=Ture, then stores the final state as the initial conditions for the next segment.
        Returns sampled history: u[:,:],t[:]."""
        nt = int(T / si)
        u, t = integrate_burgers(
            self.u, si, nt, self.dx, self.nu, t0=self.t, method=self.method
        )
        if store:
            self.u, self.t = u[-1], t[-1]
        return u, t

    def reset(self):
        self.u = self.u_init
        self.t = 0


def train_model(net, criterion, loader, optimizer):
    net.train()
    test_loss = 0
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        b_x = Variable(batch_x)  # Inputs
        b_y = Variable(batch_y)  # outputs
        if len(b_x.shape) == 1:
            prediction = torch.squeeze(
                net(torch.unsqueeze(b_x, 1))
            )  # input x and predict based on x
        else:
            prediction = net(b_x)
        loss = criterion(prediction, b_y)  # Calculating loss
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients to update weights


def test_model(net, criterion, loader, text="test"):
    net.eval()  # Evaluation mode (important when having dropout layers)
    test_loss = 0
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            b_x = Variable(batch_x)  # Inputs
            b_y = Variable(batch_y)  # outputs
            if len(b_x.shape) == 1:
                prediction = torch.squeeze(
                    net(torch.unsqueeze(b_x, 1))
                )  # input x and predict based on x
            else:
                prediction = net(b_x)
            loss = criterion(prediction, b_y)  # Calculating loss
            test_loss = test_loss + loss.data.numpy()  # Keep track of the loss
        test_loss /= len(loader)  # dividing by the number of batches
    return test_loss


def train_network(
    net,
    X_feature_train,
    X_target_train,
    X_feature_test,
    X_target_test,
    batch_size=100,
    n_epochs=1000,
):
    # Wrapper function to train all neural networks and output its training and test loss
    # Define data loaders
    torch_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_feature_train), torch.from_numpy(X_target_train)
    )
    loader_train = torch.utils.data.DataLoader(
        dataset=torch_dataset, batch_size=batch_size, shuffle=True
    )
    torch_dataset_test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_feature_test), torch.from_numpy(X_target_test)
    )
    loader_test = torch.utils.data.DataLoader(
        dataset=torch_dataset_test, batch_size=batch_size, shuffle=True
    )

    # Define and train a neural network
    criterion = torch.nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.003, amsgrad=True)
    test_loss = list()
    train_loss = list()
    for epoch in range(1, n_epochs + 1):
        train_model(net, criterion, loader_train, optimizer)
        train_loss.append(test_model(net, criterion, loader_train, text="train"))
        test_loss.append(test_model(net, criterion, loader_test, text="test"))
        print(
            "epoch = {0}, traing loss = {1:e}, test loss = {2:e}".format(
                epoch, train_loss[-1], test_loss[-1]
            ),
            end="\r",
        )
    print("\n")

    return net, train_loss, test_loss


def get_poly_features(X):
    [N, n] = X.shape
    n_feature = int(n + n + n * (n - 1) / 2)
    X_feature = np.zeros([N, n_feature])
    from itertools import combinations

    index = list(combinations(np.arange(n), 2))
    # Polynomial order: x1 + x2 + ... + x1^2 + x2^2 + ... +
    # x1*x2 + x1*x3 + ... + x2*x3 + x2*x4 + ... + x_{n-1}*x_n
    X_feature[:, 0:n] = X
    X_feature[:, n : 2 * n] = X**2
    for i in range(len(index)):
        X_feature[:, 2 * n + i] = X[:, index[i][0]] * X[:, index[i][1]]
    return X_feature


def gen_local_training_data(X, dX, filter_loc, train_size, test_size):
    Nt, K = X.shape
    n = len(filter_loc)
    X_filter = np.zeros((Nt * K, n))
    for fi in range(len(filter_loc)):
        X_filter[:, fi] = np.roll(X, filter_loc[fi], axis=1).reshape(-1)
    dX = dX.reshape(-1, 1)

    # training data
    ind = np.random.choice(Nt * K, train_size + test_size)
    X_local_train = X_filter[ind[:train_size], :]
    dX_train = dX[ind[:train_size], :]

    # test data
    X_local_test = X_filter[ind[train_size:], :]
    dX_test = dX[ind[train_size:], :]

    return X_local_train, dX_train, X_local_test, dX_test


def run_analysis(W, si, T, n_epochs=1000, Poly_init=False):
    np.random.seed(14)  # For reproducibility
    torch.manual_seed(14)  # For reproducibility

    if isinstance(W, list):
        X_true_list = []
        dX_true_list = []
        for w in W:
            X_true_temp = w.run(si, T, store=True)[0]
            X_true_list.append(X_true_temp[:-1, :])
            dX_true_list.append(X_true_temp[1:, :] - X_true_temp[:-1, :])
            del X_true_temp
        X_true = np.concatenate(X_true_list, axis=0)
        dX_true = np.concatenate(dX_true_list, axis=0)
        del X_true_list, dX_true_list
    else:
        X_true_temp = W.run(si, T, store=True)[0]
        X_true = X_true_temp[:-1, :]
        dX_true = X_true_temp[1:, :] - X_true_temp[:-1, :]
        del X_true_temp

    # Set size of training data
    train_size = 2000  # Number of training data
    test_size = 400  # Number of test data
    batch_size = 100  # Number of training data per batch

    ind = np.random.choice(X_true.shape[0], train_size + test_size)
    # Add one extra dimension to represent channels
    X_feature_train = X_true[ind[:train_size], :][:, np.newaxis, :]
    X_target_train = dX_true[ind[:train_size], :][:, np.newaxis, :]
    X_feature_test = X_true[ind[train_size:], :][:, np.newaxis, :]
    X_target_test = dX_true[ind[train_size:], :][:, np.newaxis, :]

    # Define CNN parameters
    K = X_true.shape[1]
    inout_size = [K, K]  # Input and output are both K dimensions
    kernel_size = 5  # local kernel has a width of 5 grid points
    channels = [
        2,
        4,
    ]  # number of channels of the first and second layer, both set to 1 for simplicity

    # Define a global CNN
    net_CNN_global = Net_CNN(
        inout_size=inout_size,
        kernel_size=kernel_size,
        hidden_neurons=[100, 100],
        channels=channels,
    ).double()

    # Train the global CNN using global data
    print("training global CNN\n")
    net_CNN_global, train_loss_CNN, test_loss_CNN = train_network(
        net_CNN_global,
        X_feature_train,
        X_target_train,
        X_feature_test,
        X_target_test,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )

    # Define a global ANN
    net_ANN_global = Net_ANN([K, 100, 100, K], filter_loc=[0]).double()

    # Train the global ANN using global data
    print("training global ANN\n")
    net_ANN_global, train_loss_ANN_global, test_loss_ANN_global = train_network(
        net_ANN_global,
        X_feature_train,
        X_target_train,
        X_feature_test,
        X_target_test,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )

    filter_start = -2
    filter_end = 2
    # this includes k+2, k+1, k, k-1, k-2, note that roll by +2 means looking at location k-2
    filter_loc = np.arange(filter_start, filter_end + 1)
    n = len(filter_loc)  # network input dimension

    # Generate local maps consistent with the dimension of the stencil
    X_local_train, dX_train, X_local_test, dX_test = gen_local_training_data(
        X_true, dX_true, filter_loc, train_size, test_size
    )

    # Generate polynomial features from the local training data
    X_feature_train = get_poly_features(X_local_train)
    X_feature_test = get_poly_features(X_local_test)

    net_poly = Net_ANN([X_feature_train.shape[-1], 1], filter_loc=filter_loc).double()

    # Initiate weights using simple polynomial regression
    if Poly_init:
        from sklearn import linear_model

        regr = linear_model.LinearRegression()
        regr.fit(X_feature_train, dX_train)
        net_poly.layers[0].weight.data = torch.from_numpy(regr.coef_)
        net_poly.layers[0].bias.data = torch.from_numpy(regr.intercept_)

    print("training local polynet\n")
    net_poly, train_loss_poly, test_loss_poly = train_network(
        net_poly,
        X_feature_train,
        dX_train,
        X_feature_test,
        dX_test,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )

    # Train a local ANN
    print("training local ANN\n")
    net_ANN_local = Net_ANN(
        [len(filter_loc), 10, 10, 1], filter_loc=filter_loc
    ).double()
    net_ANN_local, train_loss_ANN, test_loss_ANN = train_network(
        net_ANN_local,
        X_local_train,
        dX_train,
        X_local_test,
        dX_test,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )

    return net_ANN_global, net_CNN_global, net_ANN_local, net_poly, X_true, dX_true


def animate_Burgers(
    X_simulation_truth,
    X_simulation_global_ANN,
    X_simulation_global_CNN,
    X_simulation_local_ANN,
    X_simulation_poly,
    plot_path,
):
    from matplotlib.animation import FuncAnimation

    interval = 20
    frames = 100
    K = X_simulation_truth.shape[1]

    x = np.arange(0, K)
    fig = plt.figure(figsize=(6, 5))
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_xlim(0, K)
    ax1.set_ylim(-2, 2)
    (line0_1,) = ax1.plot([], [], lw=2, label="Ground truth")
    (line1,) = ax1.plot([], [], lw=1, label="Global ANN")
    ax1.legend(frameon=False, loc="upper right")

    ax2 = plt.subplot(2, 2, 2)
    ax2.set_xlim(0, K)
    ax2.set_ylim(-2, 2)
    (line0_2,) = ax2.plot([], [], lw=2, label="Ground truth")
    (line2,) = ax2.plot([], [], lw=1, label="Global CNN")
    ax2.legend(frameon=False, loc="upper right")

    ax3 = plt.subplot(2, 2, 3)
    ax3.set_xlim(0, K)
    ax3.set_ylim(-2, 2)
    (line0_3,) = ax3.plot([], [], lw=2, label="Ground truth")
    (line3,) = ax3.plot([], [], lw=1, label="Local ANN")
    ax3.legend(frameon=False, loc="upper right")

    ax4 = plt.subplot(2, 2, 4)
    ax4.set_xlim(0, K)
    ax4.set_ylim(-2, 2)
    (line0_4,) = ax4.plot([], [], lw=2, label="Ground truth")
    (line4,) = ax4.plot([], [], lw=1, label="Polynet")
    ax4.legend(frameon=False, loc="upper right")

    line0_list = [line0_1, line0_2, line0_3, line0_4]
    plt.tight_layout()

    def animate(i):
        for line0 in line0_list:
            line0.set_data(x, X_simulation_truth[i * interval, :])
        line1.set_data(x, X_simulation_global_ANN[i * interval, :])
        line2.set_data(x, X_simulation_global_CNN[i * interval, :])
        line3.set_data(x, X_simulation_local_ANN[i * interval, :])
        line4.set_data(x, X_simulation_poly[i * interval, :])
        return

    anim = FuncAnimation(fig, animate, frames=frames, interval=1, blit=True)
    gif_name = os.path.join(plot_path, "Burgers_simulation.gif")
    anim.save(gif_name, writer="pillow")
