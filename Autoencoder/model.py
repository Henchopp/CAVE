import torch
import torch.nn as nn


class Encoder(nn.Module):
	"""docstring for Encoder"""

	def __init__(self, enc_len):
		super(Encoder, self).__init__()

		self.conv1_1 = ConvBatchReLU(3, 16)
		self.conv1_2 = ConvBatchReLU(16, 16)
		self.maxpool1 = nn.MaxPool2d(kernel_size = 2,
		                             stride = 2)

		self.conv2_1 = ConvBatchReLU(16, 16)
		self.conv2_2 = ConvBatchReLU(16, 16)
		self.maxpool2 = nn.MaxPool2d(kernel_size = 2,
		                             stride = 2)

		self.conv3_1 = ConvBatchReLU(16, 16)
		self.conv3_2 = ConvBatchReLU(16, 4)
		self.maxpool3 = nn.MaxPool2d(kernel_size = 2,
		                             stride = 2)

		self.lin1 = nn.Linear(in_features = 8 * 8 * 4,
		                      out_features = 100,
		                      bias = True)
		self.lin1_bn = nn.BatchNorm1d(num_features = 100)
		self.lin1_relu = nn.LeakyReLU()
		self.lin2 = nn.Linear(in_features = 100,
		                      out_features = enc_len,
		                      bias = True)
		self.lin2_relu = nn.LeakyReLU()

	def forward(self, x):
		x = self.maxpool1(self.conv1_2(self.conv1_1(x)))
		x = self.maxpool2(self.conv2_2(self.conv2_1(x)))
		x = self.maxpool3(self.conv3_2(self.conv3_1(x)))

		x = x.reshape(x.shape[0], -1)
		x = self.lin1_relu(self.lin1_bn(self.lin1(x)))

		return self.lin2_relu(self.lin2(x))


class Decoder(nn.Module):
	"""docstring for Decoder"""

	def __init__(self, enc_len):
		super(Decoder, self).__init__()

		self.lin1 = nn.Linear(in_features = enc_len,
		                      out_features = 100,
		                      bias = True)
		self.lin1_bn = nn.BatchNorm1d(num_features = 100)
		self.lin1_relu = nn.LeakyReLU()

		self.lin2 = nn.Linear(in_features = 100,
		                      out_features = 8 * 8 * 4,
		                      bias = True)
		self.lin2_bn = nn.BatchNorm1d(num_features = 8 * 8 * 4)
		self.lin2_relu = nn.LeakyReLU()
		
		self.conv1 = ConvBatchReLU(4, 16)
		self.dconv1 = DeconvBatchReLU(16, 16)

		self.conv2 = ConvBatchReLU(16, 16)
		self.dconv2 = DeconvBatchReLU(16, 16)

		self.conv3 = ConvBatchReLU(16, 16)
		self.dconv3 = nn.ConvTranspose2d(in_channels = 16,
		                                 out_channels = 3,
		                                 kernel_size = 4,
		                                 stride = 2,
		                                 padding = 1,
		                                 bias = True)
		
	def forward(self, x):
		x = self.lin1_relu(self.lin1_bn(self.lin1(x)))
		x = self.lin2_relu(self.lin2_bn(self.lin2(x)))

		x = x.reshape(x.shape[0], 4, 8, 8)
		x = self.dconv1(self.conv1(x))
		x = self.dconv2(self.conv2(x))
		
		return self.dconv3(self.conv3(x))


class ConvBatchReLU(nn.Module):
	"""docstring for ConvBatchReLU"""

	def __init__(self, in_channels, out_channels):
		super(ConvBatchReLU, self).__init__()
		
		self.conv = nn.Conv2d(in_channels = in_channels,
		                      out_channels = out_channels,
		                      kernel_size = 3,
		                      stride = 1,
		                      padding = 1,
		                      padding_mode = 'replicate',
		                      bias = False)
		self.conv_bn = nn.BatchNorm2d(num_features = out_channels)
		self.relu = nn.LeakyReLU()

	def forward(self, x):
		return self.relu(self.conv_bn(self.conv(x)))


class DeconvBatchReLU(nn.Module):
	"""docstring for DeconvBatchReLU"""

	def __init__(self, in_channels, out_channels):
		super(DeconvBatchReLU, self).__init__()
		
		self.dconv = nn.ConvTranspose2d(in_channels = in_channels,
		                                out_channels = out_channels,
		                                kernel_size = 4,
		                                stride = 2,
		                                padding = 1,
		                                bias = False)
		self.dconv_bn = nn.BatchNorm2d(num_features = out_channels)
		self.relu = nn.LeakyReLU()

	def forward(self, x):
		return self.relu(self.dconv_bn(self.dconv(x)))
		


###