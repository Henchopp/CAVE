from model import AutoEncoder
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
import torchvision
import torch.nn as nn
import numpy as np
import torch
import time
import copy
import os
from PIL import Image
from CAVE.cave import CAVE

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

class ImageNetData(Dataset):

    def __init__(self, directory):

        self.directory = directory
        self.file_names = os.listdir(directory)
        self.crop = torchvision.transforms.RandomCrop(64)

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        file_path = os.path.join(self.directory, self.file_names[idx])

        img = torchvision.io.read_image(file_path).float() / 255

        return self.crop(img)

def train(epochs = 1000, cave = False):

    train = ImageNetData("/home/prs5019/cave/image_net/train")
    valid = ImageNetData("/home/prs5019/cave/image_net/valid")
    test = ImageNetData("/home/prs5019/cave/image_net/test")

    train_loader = DataLoader(train, batch_size = 1536, shuffle = True, num_workers = 16)
    valid_loader = DataLoader(valid, batch_size = 1536, shuffle = True, num_workers = 16)
    test_loader = DataLoader(test, batch_size = 10, shuffle = False, num_workers = 1)

    model = AutoEncoder(use_cave = cave)

    model = model.to(device)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = 1.0e-4, betas = (0.9, 0.999))

    min_loss = np.inf
    n_no_decrease = 0

    train_t_losses = []
    valid_t_losses = []
    times = []

    for e in range(epochs):

        train_losses = []

        start = time.time()

        for feat in train_loader:

            feat = feat.float().to(device)

            optimizer.zero_grad()

            loss = F.mse_loss(model(feat), feat) # getting mean squared error loss
            train_losses.append(loss.item())
            loss.backward() # backwards sweep

            optimizer.step() # adjusting model parameters

        end = time.time()
        times.append(end - start)

        train_t_losses.append(sum(train_losses) / len(train_losses))

        valid_losses = []

        with torch.no_grad():

            for feat in valid_loader:
                feat = feat.to(device)

                loss = F.mse_loss(model(feat), feat)

                valid_losses.append(loss.item())

        valid_t_losses.append(sum(valid_losses) / len(valid_losses))

        if(valid_t_losses[-1] < min_loss):
            min_loss = valid_t_losses[-1]
            max_state_dict = copy.deepcopy(model).state_dict()

            n_no_decrease = 0
        else:
            n_no_decrease += 1

        if(n_no_decrease > 300):
            break

        print(f"Epoch {e} | Valid Loss {valid_t_losses[-1]} | Train Loss {train_t_losses[-1]}")

    test_losses = []
    input_mean = []
    output_mean = []

    to_pil = ToPILImage()

    with torch.no_grad():

        for idx, feat in enumerate(test_loader):

            for im in range(feat.detach().cpu().shape[0]):
                input = to_pil(feat.detach().cpu()[im])
                input.save(f"/home/prs5019/cave/inpainting/cave/test_inputs/{idx}_{im}.jpeg")
                input_mean.append(feat.detach().cpu().mean())

            feat = feat.to(device)

            decoded = model(feat)

            for im in range(decoded.detach().cpu().shape[0]):
                output = to_pil(decoded.detach().cpu()[im])
                output.save(f"/home/prs5019/cave/inpainting/cave/test_outputs/{idx}_{im}.jpeg")
                output_mean.append(decoded.detach().cpu().mean())

            loss = F.mse_loss(decoded, feat)

            test_losses.append(loss.item())

    # saving
    torch.save(max_state_dict, "/home/prs5019/cave/inpainting/cave/model")
    # saving losses
    np.save("/home/prs5019/cave/inpainting/cave/valid_losses", np.array(valid_t_losses))
    np.save("/home/prs5019/cave/inpainting/cave/train_losses", np.array(train_t_losses))
    np.save("/home/prs5019/cave/inpainting/cave/test_losses", np.array(test_losses))
    # saving means and mean divergence
    np.save("/home/prs5019/cave/inpainting/cave/in_mean", np.array(input_mean))
    np.save("/home/prs5019/cave/inpainting/cave/out_mean", np.array(output_mean))
    np.save("/home/prs5019/cave/inpainting/cave/mean_divergence", np.abs(np.array(input_mean) - np.array(output_mean)))
    # saving times
    np.save("/home/prs5019/cave/inpainting/cave/epoch_times", np.array(times))

if(__name__ == "__main__"):
    train(cave = False)
