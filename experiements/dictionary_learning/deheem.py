import torch.nn.functional as F
import numpy as np
from CAVE.cave import CAVE
from CAVE.utils.cave_base_functions import Sigmoid
import torch
import torchvision
from PIL import Image
import pandas as pd
import h5py
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

def train(xrf_path, thresh, M, epochs = 100):
    print('Reading data...')
    # ================ reading in data ====================
    with h5py.File(xrf_path, "r") as hf:
        xrf = torch.from_numpy(np.array(hf["data"][:], dtype = np.float64)) # reaing
        xrf = xrf.permute(0,2,1).float()
        xrf[xrf < 0.0] = 0.0
        xrf = xrf.round()

    inds = xrf.sum(dim = [1,2]) > thresh # getting indices from sum dist
    xrf = xrf[inds,:,:]
    xrf = xrf.reshape(xrf.shape[0], xrf.shape[1] * xrf.shape[2]).to(device)

    # ================== optimization initializations =================

    global_min = F.poisson_nll_loss(xrf, xrf, log_input = False).item() # getting global min to make minimum loss 0 later

    D_data = torch.from_numpy(np.loadtxt("/home/prs5019/cave/cave_data/D_raster0,05_all.csv", dtype = np.float64)[inds.numpy()]).to(device)

    with h5py.File("/home/prs5019/cave/cave_data/A_raster0,05_all.h5") as hf:

        A_data = torch.from_numpy(np.array(hf["data"][:], dtype = np.float64)).to(device)

    D = torch.nn.Parameter(data = D_data, requires_grad = True)
    A = torch.nn.Parameter(data = A_data, requires_grad = True)

    A_v = A.view(A.shape[0], 578, 673)

    optimizer = torch.optim.Adam([D, A], lr = 1.0e-3, betas = (0.9, 0.999))

    cave = CAVE(func = Sigmoid(), n_step_nm = 15, n_step_gd = 5).to(device)

    # ================= smoothing initializations ======================

    image = torchvision.transforms.ToTensor()(Image.open("/home/prs5019/cave/cave_data/deheem.png"))
    t = 0.05

    # Vertical adaptive weights
    tv_adap_w_r = ((image[:,:-1,:] - image[:,1:,:]) ** 2).sum(dim = 0, keepdim = True)
    tv_adap_w_r = (-16.0 * tv_adap_w_r).exp() / t
    tv_adap_w_r = tv_adap_w_r.to(device)

    # Horizontal adaptive weights
    tv_adap_w_c = ((image[:,:,:-1] - image[:,:,1:]) ** 2).sum(dim = 0, keepdim = True)
    tv_adap_w_c = (-16.0 * tv_adap_w_c).exp() / t
    tv_adap_w_c = tv_adap_w_c.to(device)

    # ================= data collection initializations ================

    losses = []

    min_loss = None
    min_D = None
    min_A = None

    last_10 = 0
    prev_last_10 = 1


    for e in range(epochs):

        optimizer.zero_grad() # zeroing gradients

        A_cave = cave(A, low = 0, high = 1, mean = 5 / 37, var = 40 / 333, sparse = True)
        output = torch.matmul(F.relu(D), F.relu(A) * A_cave)
	
        # ============ smoothing loss ==============
        l_tv = 0.1
        tv_r = (tv_adap_w_r * ((A_v[:,:-1,:] - A_v[:,1:,:]) ** 2)).mean()
        tv_c = (tv_adap_w_c * ((A_v[:,:,:-1] - A_v[:,:,1:]) ** 2)).mean()

        loss = F.poisson_nll_loss(output, xrf, log_input = False) + l_tv * (tv_r + tv_c) # getting loss

        loss.backward() # calculating gradients

        optimizer.step() # updating weights based on gradients

        losses.append(loss.item())

        # ================ early stopping and lr adjustment ====================

        if(min_loss == None or loss.item() - global_min < min_loss):
            min_loss = loss.item() - global_min
            min_D = D
            min_A = A


        print(f"Loss {loss.item() - global_min} | Epoch: {e} | Mean: {A_cave.mean()} | Var: {A_cave.var()}")

        # decreasing learning rate when threshold reached
        if(loss.item() - global_min <= 1.186):
            for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"]# * 0.1

        if(e != 0 and e % 10 == 0):

            # see if we should break
            if(100 * (1 - last_10 / prev_last_10) < 0.001 and e != 10):
                break

            prev_last_10 = last_10
            last_10 = 0

        else:
            last_10 += (loss.item() - global_min) / 10

    torch.save(min_D, "/home/prs5019/cave/min_D")
    torch.save(min_A, "/home/prs5019/cave/min_A")


if(__name__ == "__main__"):
    train("/home/prs5019/cave/cave_data/deheem_raster0,05.h5", 0, 37, 50_000)
