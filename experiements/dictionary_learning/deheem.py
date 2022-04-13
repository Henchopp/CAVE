import torch.nn.functional as F
import numpy as np
from CAVE.cave import CAVE
from CAVE.utils.cave_base_functions import Sigmoid
import torch
import h5py
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, torch.cuda.device_count())

def train(xrf_path, thresh, M, epochs = 100):
    print('Reading data...')

    with h5py.File(xrf_path, "r") as hf:
        xrf = torch.from_numpy(np.array(hf["data"][:], dtype = np.float64)) # reaing
        xrf = xrf.permute(0,2,1).float()
        xrf[xrf < 0.0] = 0.0
        xrf = xrf.round()

    inds = xrf.sum(dim = [1,2]) > thresh # getting indices from sum dist
    xrf = xrf[inds,:,:]
    xrf = xrf.reshape(xrf.shape[0], xrf.shape[1] * xrf.shape[2]).to(device)

    # print(F.poisson_nll_loss(xrf, xrf, log_input = False).item())
    D = torch.nn.Parameter(data = torch.rand(xrf.shape[0], M, device = device), requires_grad = True)
    A = torch.nn.Parameter(data = torch.rand(M, xrf.shape[1], device = device), requires_grad = True)

    optimizer = torch.optim.Adam([D, A], lr = 1.0e-1, betas = (0.9, 0.999))

    cave = CAVE(func = Sigmoid()).to(device)

    losses = []

    min_loss = None
    min_D = None
    min_A = None
    min_index = 0

    for e in range(epochs):

        optimizer.zero_grad() # zeroing gradients
        output = torch.matmul(F.softplus(D), A * cave(A, low = 0, high = 1, mean = 0.1, var = 0.1, sparse = True))

        loss = F.poisson_nll_loss(output, xrf, log_input = False) # getting loss

        loss.backward() # calculating gradients

        optimizer.step() # updating weights based on gradients

        losses.append(loss.item())
        print(loss.item() + 17.013126373291016)
        if(min_loss == None or loss.item() + 17.013126373291016 < min_loss):
            min_loss = loss.item() + 17.013126373291016
            min_D = D
            min_A = A
            min_index = e

        print(f"Loss {loss.item() + 17.013126373291016} | Epoch: {e}")

        if(e - min_index > 100):
            break


if(__name__ == "__main__"):
    train("/home/prs5019/cave/deheem_orig.h5", 10, 5000)
