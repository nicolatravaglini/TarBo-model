from constants import *
from model import TarBoModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


class Trainer:
	def __init__(self, training_set_dir, test_set_directory):
		# Device definition
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# Dataset definition
		all_transforms = transforms.Compose([transforms.Resize((IMG_RESIZE_WIDTH, IMG_RESIZE_HEIGHT)), transforms.ToTensor()])
		training_set = ImageFolder(training_set_dir, all_transforms)
		test_set = ImageFolder(test_set_directory, all_transforms)
		self.training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
		self.test_loader = DataLoader(test_set, shuffle=True)
		# Model definition
		self.model = TarBoModel().to(self.device)
		self.criterion = nn.NLLLoss()
		self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

	def train(self):
		print("--- TRAINING ---")
		for epoch in range(NUM_EPOCHS):
			for images, labels in self.training_loader:
				images = images.to(self.device)
				labels = labels.to(self.device)
				# Forward
				outputs = self.model(images)
				loss = self.criterion(outputs, labels)
				# Backward
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item()}")
		self.save()

	def test(self):
		print("--- TESTING ---")
		with torch.no_grad():
			self.model.eval()
			correct = 0
			for images, labels in self.test_loader:
				outputs = self.model(images)
				_, predicted = torch.max(outputs.data, 1)
				correct += (predicted == labels).item()
			# Print accuracy
			accuracy = 100 * correct / len(self.test_loader)
			print(f"Accuracy: {accuracy}%")

	def save(self):
		torch.save(self.model.state_dict(), "model.pt")


if __name__ == "__main__":
	trainer = Trainer("dataset/training_set", "dataset/test_set")
	trainer.train()
	trainer.test()
