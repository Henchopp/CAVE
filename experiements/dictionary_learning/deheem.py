import numpy as np
import torch
import h5py
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, torch.cuda.device_count())

def train(xrf_path, thresh, M):
    print('Reading data...')

    with h5py.File(xrf_path, "r") as hf:
        xrf = torch.from_numpy(np.array(hf["data"][:], dtype = np.float64)) # reaing
        xrf = xrf.permute(0,2,1).unsqueeze(1).float()
        xrf[xrf < 0.0] = 0.0
        xrf = xrf.round()

    inds = xrf.sum(dim = [1,2,3]) > thresh # getting indices from sum dist
    xrf = xrf[inds,:,:,:]
    xrf = torch.transpose(torch.transpose(xrf, 0, 1)[0], 0, 2).shape

    D = torch.nn.Parameter(data = torch.rand(806, M), requires_grad = True)

    A = torch.nn.Parameter(data = torch.rand(M, 578 * 673), requires_grad = True)



if(__name__ == "__main__"):
    train("/home/prs5019/cave/deheem_orig.h5", 10, 100)
