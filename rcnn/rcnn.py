from constants import *
import torch
from torchvision import transforms
import selective_search
from model.model import TarBoModel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Box:
	def __init__(self, x1, y1, x2, y2, predicted_class, probability):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.predicted_class = predicted_class
		self.probability = probability


class RCNN:
	def __init__(self):
		self.model = TarBoModel()
		self.model.load_state_dict(torch.load("../model/model.pt"))
		self.model.eval()

	def non_max_suppression(self, boxes):
		# Per ogni classe
		for prediction_class in OUTPUTS.values():
			class_boxes = [box for box in boxes if box.predicted_class == prediction_class]

			if len(class_boxes) > 0:
				# Seleziona la box con probabilità più alta
				highest_probability_box = class_boxes[0]
				for box in class_boxes:
					if box.probability > highest_probability_box.probability:
						highest_probability_box = box

				# Elimina tutte le box che entrano o collidono con quella di maggiore probabilità
				# (non tutte le restanti perché ci sono due carte uguali nel gioco)
				highest_width = (highest_probability_box.x2-highest_probability_box.x1)//2
				highest_height = (highest_probability_box.y2-highest_probability_box.y1)//2
				highest_center_x, highest_center_y = highest_probability_box.x1+highest_width, highest_probability_box.y1+highest_height
				for box in class_boxes:
					if box != highest_probability_box:
						box_center_x, box_center_y = box.x1+(box.x2-box.x1)//2, box.y1+(box.y2-box.y1)//2
						if abs(box_center_x-highest_center_x) < highest_width*2 or abs(box_center_y-highest_center_y) < highest_height*2:
							boxes.remove(box)

	def false_max_suppression(self, boxes):
		to_delete = []
		for box in boxes:
			if box.probability <= 30:
				to_delete.append(box)

		for prediction_class in OUTPUTS.keys():
			class_boxes = [box for box in boxes if box.predicted_class == prediction_class]
			if len(class_boxes) > 0:
				# Seleziona la box con probabilità più alta
				highest_probability_box = class_boxes[0]
				for box in class_boxes:
					if box.probability > highest_probability_box.probability:
						highest_probability_box = box

				# Rimuovi tutte le altre box
				for box in class_boxes:
					if box != highest_probability_box and box not in to_delete:
						to_delete.append(box)

		for box in to_delete:
			boxes.remove(box)

	def classify(self, image):
		# Individua le regioni d'interesse (RoI)
		boxes = selective_search.selective_search(image, mode="single", random_sort=True)
		filtered_boxes = selective_search.box_filter(boxes, min_size=50, topN=80)

		# Per ogni regione ridimensionala e predici la classe con il modello
		boxes = []
		box_imgs = []
		for x1, y1, x2, y2 in filtered_boxes:
			box_img = image[y1:y2, x1:x2]
			if box_img.size > 0:
				# Adatto il box al modello di CNN
				all_transforms = transforms.Compose([
					transforms.ToTensor(),
					transforms.Resize((IMG_RESIZE_WIDTH, IMG_RESIZE_HEIGHT)),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
				])
				box_img = all_transforms(box_img)
				box_imgs.append(box_img)
				box_img = box_img.unsqueeze(0)

				# Ottengo la classe, faccio NMS e la salvo (solo se non è lo sfondo)
				output = self.model(box_img)
				probability, predicted = torch.max(output.data, 1)
				predicted_class = list(OUTPUTS)[predicted.item()]
				if predicted_class != "sfondo":
					boxes.append(Box(x1, y1, x2, y2, predicted_class, probability.item()))

		fig, axes = plt.subplots(nrows=10, ncols=8, figsize=(12, 10))
		for box, ax in zip(box_imgs, axes.flat):
			# Mostra l'immagine
			ax.imshow(box.permute(1, 2, 0))
			ax.axis('off')
		plt.tight_layout()
		plt.show()

		# Applico NMS
		# self.non_max_suppression(boxes)
		self.false_max_suppression(boxes)

		fig, ax = plt.subplots(figsize=(6, 6))
		ax.imshow(image)
		for box in boxes:
			bbox = mpatches.Rectangle((box.x1, box.y1), (box.x2 - box.x1), (box.y2 - box.y1), fill=False, edgecolor='red', linewidth=1)
			ax.add_patch(bbox)
			plt.text(box.x1, box.y1, box.predicted_class+"-"+str(int(box.probability)), color="yellow")
		plt.axis('off')
		plt.show()

		# Salva e restituisci i risultati
		return [box.predicted_class for box in boxes]
