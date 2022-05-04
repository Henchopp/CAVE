import torch
from torch import nn
import torch.nn.functional as F
from CAVE.cave import CAVE
from CAVE.utils.cave_base_functions import Sigmoid


class BatchConvReLU(nn.Module):

    def __init__(self, inp, out, ksize = 3):
        super(BatchConvReLU, self).__init__()

        self.conv = nn.Conv2d(
            in_channels = inp,
            out_channels = out,
            kernel_size = ksize,
            stride = 1,
            padding = 1,
            # padding_mode = "replicate",
            bias = False
        )

        self.norm = nn.BatchNorm2d(num_features = out)

    def forward(self, x):

        return self.norm(F.relu(self.conv(x)))


class BatchDeconvReLU(nn.Module):

    def __init__(self, inp, out, ksize = 4):
        super(BatchDeconvReLU, self).__init__()

        self.dconv = nn.ConvTranspose2d(
            in_channels = inp,
            out_channels = out,
            kernel_size = ksize,
            stride = 2,
            padding = 1,
            bias = 4
        )

        self.norm = nn.BatchNorm2d(num_features = out)

    def forward(self, x):

        return self.norm(F.relu(self.dconv(x)))


class Encoder(nn.Module):

    def __init__(self, encoding_space = 100):
        super(Encoder, self).__init__()

        # layer 1
        self.conv1 = BatchConvReLU(3, 16)
        # layer 2
        self.conv2 = BatchConvReLU(16, 32)
        self.mpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # layer 3
        self.conv3 = BatchConvReLU(32, 64)
        # layer 4
        self.conv4 = BatchConvReLU(64, 64)
        # layer 5
        self.conv5 = BatchConvReLU(64, 16)
        self.mpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # ======= MLP =======

        self.fc1 = nn.Linear(4096, 1024, bias = True)
        self.fc2 = nn.Linear(1024, encoding_space, bias = True) # should output vector in encoded space

    def forward(self, x):

        # === convolving ===
        x = self.conv1(x)
        x = self.mpool1(self.conv2(x))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mpool2(self.conv5(x))

        # === flattening ===
        x = x.reshape(x.shape[0], -1)

        # === mlp ===
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class Decoder(nn.Module):

    def __init__(self, encoding_space = 100):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(encoding_space, 1024) # should take in encoded vector
        self.bn1 = nn.BatchNorm1d(num_features = 1024)

        self.fc2 = nn.Linear(1024, 4096)
        self.bn2 = nn.BatchNorm1d(num_features = 4096)

        self.conv1 = BatchConvReLU(64, 16)
        self.dconv1 = BatchDeconvReLU(16, 16)

        self.conv2 = BatchConvReLU(16, 16)
        self.dconv2 = BatchDeconvReLU(16, 16)

        self.conv3 = BatchConvReLU(16, 16)
        self.dconv3 = nn.ConvTranspose2d(in_channels = 16,
		                                 out_channels = 3,
		                                 kernel_size = 4,
		                                 stride = 2,
		                                 padding = 1,
		                                 bias = True)

    def forward(self, x):

        # === mlp ===
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        # === reshape ===
        x = x.reshape(x.shape[0], 64, 8, 8)

        # === convolve ===
        x = self.dconv1(self.conv1(x))
        x = self.dconv2(self.conv2(x))

        return self.dconv3(self.conv3(x))

class AutoEncoder(nn.Module):

    def __init__(self, encoding_space = 100, use_cave = False):
        super(AutoEncoder, self).__init__()

        if(use_cave == True):
            encoding_space = encoding_space - 2

        self.cave = CAVE(func = Sigmoid()) if use_cave == True else None

        self.encoder = Encoder(encoding_space = encoding_space)
        self.decoder = Decoder(encoding_space = encoding_space)

    def forward(self, x):

        if(self.cave is not None):
            output = self.decoder(self.encoder(x))
            output = self.cave(output,
                            low = 0.0,
                            high = 1.0,
                            mean = x.mean(dim = [1, 2, 3], keepdim = True),
                            var = x.var(dim  = [1, 2, 3], keepdim = True, unbiased = False), 
                            dim = [1, 2, 3],
                            unbiased = False)
        else:
            output = self.decoder(self.encoder(x))

        return output

    def encode(self, x):

        return self.encoder(x)

    def decode(self, x):

        return self.decoder(x)
