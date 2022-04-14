import torch.nn.functional as F
import numpy as np
from CAVE.cave import CAVE
from CAVE.utils.cave_base_functions import Sigmoid
import torch
import h5py
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

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

    global_min = F.poisson_nll_loss(xrf, xrf, log_input = False).item() # getting global min to make minimum loss 0 later

    D = torch.nn.Parameter(data = torch.rand(xrf.shape[0], M, device = device), requires_grad = True)
    A = torch.nn.Parameter(data = torch.rand(M, xrf.shape[1], device = device), requires_grad = True)

    optimizer = torch.optim.Adam([D, A], lr = 1.0, betas = (0.9, 0.999))

    cave = CAVE(func = Sigmoid(), n_step_nm = 15, n_step_gd = 5).to(device)

    losses = []

    min_loss = None
    min_D = None
    min_A = None

    last_10 = 0
    prev_last_10 = 1

    for e in range(epochs):

        optimizer.zero_grad() # zeroing gradients
        output = torch.matmul(F.softplus(D), F.relu(A) * cave(A, low = 0, high = 1, mean = 0.1, var = 0.1, sparse = True))

        loss = F.poisson_nll_loss(output, xrf, log_input = False) # getting loss

        loss.backward() # calculating gradients

        optimizer.step() # updating weights based on gradients

        losses.append(loss.item())

        # ================ early stopping and lr adjustment ====================

        if(min_loss == None or loss.item() - global_min < min_loss):
            min_loss = loss.item() - global_min
            min_D = D
            min_A = A


        print(f"Loss {loss.item() - global_min} | Epoch: {e}")

        # decreasing learning rate when threshold reached
        if(loss.item() - global_min <= 0.35):
            for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * 0.1

        if(e != 0 and e % 10 == 0):

            # see if we should break
            if(100 * (1 - last_10 / prev_last_10) < 0.1 and e != 10):
                break

            prev_last_10 = last_10
            last_10 = 0

        else:
            last_10 += (loss.item() - global_min) / 10

    torch.save(min_D, "/home/prs5019/cave/min_D")
    torch.save(min_A, "/home/prs5019/cave/min_A")


if(__name__ == "__main__"):
    train("/home/prs5019/cave/deheem_orig.h5", 10, 100, 50_000)
