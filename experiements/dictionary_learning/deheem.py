import torch
import h5py
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(xrf_path, thresh):
	print('Reading data...')
    
    with h5py.File(xrf_path, "r") as hf:
		xrf = torch.from_numpy(hf["data"][:]) # reaing
		xrf = xrf.permute(0,2,1).unsqueeze(1).float()
		xrf[xrf < 0.0] = 0.0
		xrf = xrf.round()

​    inds = xrf.sum(dim = [1,2,3]) > thresh # getting indices from sum dist
    xrf = xrf[inds,:,:,:]
    print(xrf.shape)

if(__name__ == "__main__"):
    train("/cave/deheem_orig.h5", 10)
