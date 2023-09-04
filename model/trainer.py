from constants import *
from model import TarBoModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
import wandb


class Trainer:
	def __init__(self, training_set_dir, test_set_directory):
		wandb.init()
		# Device definition
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(self.device)

		# Dataset definition
		training_transforms = transforms.Compose([
			transforms.Resize((IMG_RESIZE_WIDTH, IMG_RESIZE_HEIGHT)),
			# transforms.RandomResizedCrop(IMG_RESIZE_WIDTH),
			# transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(degrees=10),
			transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		test_transforms = transforms.Compose([
			transforms.Resize((IMG_RESIZE_WIDTH, IMG_RESIZE_HEIGHT)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		training_set = ImageFolder(training_set_dir, training_transforms)
		test_set = ImageFolder(test_set_directory, test_transforms)
		print(training_set.class_to_idx)
		self.training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
		self.test_loader = DataLoader(test_set, shuffle=True)

		# Model definition
		self.model = TarBoModel().to(self.device)
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

	def train(self):
		print("--- TRAINING ---")
		for epoch in range(NUM_EPOCHS):
			wandb.log({"epoch": epoch})
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

				wandb.log({"loss": loss.item()})
			print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item()}")
		self.save()

	def test(self):
		print("--- TESTING ---")
		with torch.no_grad():
			self.model.eval()
			correct = 0
			for images, labels in self.test_loader:
				images = images.to(self.device)
				labels = labels.to(self.device)
				outputs = self.model(images)
				_, predicted = torch.max(outputs.data, 1)
				correct += (predicted == labels).item()

			# Print accuracy
			accuracy = 100 * correct / len(self.test_loader)
			print(f"Accuracy: {accuracy}%")

	def show_training_images(self):
		# Ottieni una batch d'immagini ed etichette dal training loader
		images, labels = next(iter(self.training_loader))
		images = images.numpy()
		labels = labels.numpy()

		# Visualizza le immagini e le etichette
		fig, axes = plt.subplots(nrows=4, ncols=12, figsize=(12, 4))

		for i, ax in enumerate(axes.flat):
			# Mostra l'immagine
			ax.imshow(images[i].transpose(1, 2, 0))
			ax.axis('off')

			# Mostra l'etichetta
			label = list(OUTPUTS)[labels[i]]
			ax.set_title(label)

		# Mostra il plot
		plt.tight_layout()
		plt.show()

	def save(self):
		torch.save(self.model.state_dict(), "model.pt")


def train():
	trainer = Trainer("dataset/training_set", "dataset/test_set")
	trainer.show_training_images()
	trainer.train()
	trainer.test()


if __name__ == "__main__":
	train()
	"""
	wandb.login(key="89fa9192625ce23a92c7e9b2459f2f7e61823a51")
	sweep_conf = {
		"method": "bayes",
		"metric": {
			"goal": "minimize",
			"name": "loss"
		},
		"parameters": {
			"batch_size": {
				"min": 24,
				"max": 480
			},
			"epochs": {
				"min": 2,
				"max": 6
			},
			"lr": {
				"min": 0.0001,
				"max": 0.003
			}
		}
	}
	sweep_id = wandb.sweep(sweep=sweep_conf, project="TarBo")
	wandb.agent(sweep_id, function=train, count=30)
	"""

