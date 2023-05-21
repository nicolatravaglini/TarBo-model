from constants import *
import torch.nn as nn


class TarBoModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.rel1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.rel2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.lin = nn.Linear(16*(IMG_RESIZE_WIDTH//4)*(IMG_RESIZE_WIDTH//4), NUM_CLASSES)

	def forward(self, x):
		x = self.pool1(self.rel1(self.conv1(x)))
		x = self.pool2(self.rel2(self.conv2(x)))
		x = x.reshape(x.shape[0], -1)
		x = self.lin(x)
		return x
