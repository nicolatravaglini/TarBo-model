from ..constants import *
import torch.nn as nn


class TarBoModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.convo = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

			nn.Flatten()
		)

		self.classifier = nn.Sequential(
			nn.Linear(64 * (IMG_RESIZE_WIDTH // 4) * (IMG_RESIZE_HEIGHT // 4), 128),
			nn.Dropout(0.6),
			nn.ReLU(),
			nn.Linear(128, NUM_CLASSES)
		)

	def forward(self, x):
		x = self.convo(x)
		x = self.classifier(x)
		return x
