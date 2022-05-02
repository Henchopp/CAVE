from model import AutoEncoder
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

class ImageNetData(Dataset):

    def __init__(self, directory):

        self.directory = directory
        self.file_names = os.listdir(directory)

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        file_path = os.path.join(self.directory, self.file_names[idx])

        img = Image.open(file_path).convert("RGB").resize((335, 500))
        np_img = np.array(img)
        img.close()

        return torch.from_numpy(np_img).permute(2, 0, 1)

def train(epochs = 100):

    train = ImageNetData("/home/prs5019/cave/image_net/train")
    valid = ImageNetData("/home/prs5019/cave/image_net/valid")

    train_loader = DataLoader(train, batch_size = 32, shuffle = True, num_workers = 16)
    valid_loader = DataLoader(valid, batch_size = 32, shuffle = True, num_workers = 16)

    model = AutoEncoder()

    model = model.to(device)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = 1.0e-3, betas = (0.9, 0.999))


    for e in range(epochs):

        for feat in train_loader:

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

if(__name__ == "__main__"):
    train()
