from model import AutoEncoder
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn as nn
import numpy as np
import torch
import copy
import os
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

def train(epochs = 100, cave = False):

    train = ImageNetData("/home/prs5019/cave/image_net/train")
    valid = ImageNetData("/home/prs5019/cave/image_net/valid")

    train_loader = DataLoader(train, batch_size = 4096, shuffle = True, num_workers = 16)
    valid_loader = DataLoader(valid, batch_size = 4096, shuffle = True, num_workers = 16)

    model = AutoEncoder(use_cave = cave)

    model = model.to(device)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = 1.0e-4, betas = (0.9, 0.999))

    min_loss = np.inf
    n_no_decrease = 0

    for e in range(epochs):

        for feat in train_loader:
            # print(feat.float().detach().cpu().numpy().tolist())
            feat = feat.float().to(device)

            optimizer.zero_grad()

            loss = F.mse_loss(model(feat), feat) # getting mean squared error loss

            loss.backward() # backwards sweep

            optimizer.step() # adjusting model parameters

        valid_losses = []

        with torch.no_grad():

            for feat in valid_loader:
                feat = feat.to(device)

                loss = F.mse_loss(model(feat), feat)

                valid_losses.append(loss.item())

        if(sum(valid_losses) / len(valid_losses) < min_loss):
            min_loss = sum(valid_losses) / len(valid_losses)
            max_state_dict = copy.deepcopy(model).state_dict()
        else:
            n_no_decrease += 1

        if(n_no_decrease > 9):
            break

        print(f"Epoch {e} | Valid Loss {sum(valid_losses) / len(valid_losses)}")


    # saving
    torch.save(max_state_dict, "/home/prs5019/cave/inpainting")

if(__name__ == "__main__"):
    train(cave = True)
