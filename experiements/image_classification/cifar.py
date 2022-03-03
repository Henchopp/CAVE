import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 2 convolution layers
        self.conv1 = nn.Conv2D(in_channels = 3, out_channels = 32, kernel_size = 3) # 32 kernels
        self.conv2 = nn.Conv2D(in_channels = 32, out_channels = 64, kernel_size = 3) # 2 kernels
        # normalization and dropout
        self.norm = nn.BatchNorm2d(num_features = 64)
        self.do = nn.Dropout(0.2)
        # fully connectec layers
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

         x = self.conv1(x)
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

def get_data_loader(batch_size = 32, download = False):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # scale imported image
        transforms.ToTensor()]
    )

    data = datasets.CIFAR100(root = "/cave/cifar", download = download, transform = transform)

    data_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = True)

    return data_loader

def train(epochs = 100):

    model = SimpleCNN()

    model.train()

    data_loader = get_data_loader()

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

    for e in range(epochs):

        for batch_indx, (feat, class) in enumerate(data_loader):
            feat, class = feat.to(device), class.to(device)

            optimizer.zero_grad()

            output = model(feat)

            loss = F.cross_entropy(output, class)

            loss.backward()

            optimizer.step()

    return model

train()
