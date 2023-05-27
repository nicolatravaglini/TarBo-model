from constants import *
import torch.nn as nn


class TarBoModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.lin1 = nn.Linear(64 * (IMG_RESIZE_WIDTH // 4) * (IMG_RESIZE_HEIGHT // 4), 128)
		self.rel = nn.ReLU()
		self.lin2 = nn.Linear(128, NUM_CLASSES)

	def forward(self, x):
		x = self.pool1(self.conv2(self.conv1(x)))
		x = self.pool2(self.conv4(self.conv3(x)))
		x = x.reshape(x.shape[0], -1)
		x = self.lin2(self.rel(self.lin1(x)))
		return x
