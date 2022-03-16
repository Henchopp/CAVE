import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision import datasets, transforms
from CAVE.cave import CAVE
from CAVE.utils.cave_base_functions import Sigmoid

# torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 2 convolution layers
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3) # 32 kernels
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3) # 2 kernels
        # normalization and dropout
        self.norm = nn.BatchNorm2d(num_features = 64)
        self.do = nn.Dropout(0.2)
        # fully connectec layers
        self.fc1 = nn.Linear(774400, 128)
        self.fc2 = nn.Linear(128, 100)
        # setting CAVE
        self.cave = CAVE(func = Sigmoid(), n_step_nm = 7, n_step_gd = 0)

    def forward(self, x):

        x = self.conv1(x)
        if(torch.any(torch.isnan(x)).item() == True):
            print("nan found")
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.norm(x)
        x = self.do(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim = 1)
        """
        output = self.cave(x, low = 0.0, high = 1.0, mean = 1e-2, var = None , sparse = False, dim = 1, unbiased = False)
        
        output = output + 1e-20
        output = output / output.sum(dim = 1, keepdim=True)
        output = output.log()
        """
        return output

def get_data_loader(batch_size = 600, download = False, train = True):
    
    loaders = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # scale imported image
        transforms.ToTensor()]
    )

    data = datasets.CIFAR100(root = "/cave/cifar", download = download, transform = transform, train = train)
    
    if(train == True):
        data = list(random_split(data, [40000, 10000]))
    else:
        data = [data]
    
    for data_set in data:

        loaders.append(
                torch.utils.data.DataLoader(data_set, batch_size = batch_size, shuffle = True, num_workers = 8)
        )

    return loaders

def train(epochs = 100):

    model = SimpleCNN()

    model.to(device)

    model.train()

    train, val = get_data_loader()
    test = get_data_loader(train = False)[0]

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, betas = (0.9, 0.999))

    for e in range(epochs):

        loss_hist = []

        for batch_indx, (feat, label) in enumerate(train):
            # print(feat)
            feat, label = feat.to(device), label.to(device)

            optimizer.zero_grad()

            output = model(feat)

            loss = F.nll_loss(output, label)

            loss_hist.append(loss.item())

            loss.backward()

            optimizer.step()

        # ======== validation =========
        val_loss = 0
        correct = 0
        wrong = 0
        with torch.no_grad():

            for image, label in val:
                image = image.to(device)
                label = label.to(device)

                outputs = model(image)

                val_loss += F.nll_loss(outputs, label).item()
                correct += len((outputs.argmax(dim = 1) == label).nonzero())
                wrong += 600 - len((outputs.argmax(dim = 1) == label).nonzero())


        print(f"Epoch: {e} | Train Loss: {sum(loss_hist) / len(loss_hist)} | Val Loss: {val_loss / len(val)} | Val Acc: {correct / (correct + wrong)}")

    # ======== testing =========
    test_loss = 0
    correct = 0
    wrong = 0
    with torch.no_grad():

        for image, label in test:
            image = image.to(device)
            label = label.to(device)

            outputs = model(image)

            test_loss += F.nll_loss(outputs, label).item()
            correct += len((outputs.argmax(dim = 1) == label).nonzero())
            wrong += 600 - len((outputs.argmax(dim = 1) == label).nonzero())

    print(f"Test Loss: {test_loss / len(test)} | Test Acc: {correct / (correct + wrong)}")

    return model

model = train(10)
