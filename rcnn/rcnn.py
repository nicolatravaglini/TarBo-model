from constants import *
import torch
from torchvision import transforms
import selective_search
from model.model import TarBoModel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class RCNN:
	def __init__(self):
		self.model = TarBoModel()
		self.model.load_state_dict(torch.load("../model/model.pt"))
		self.model.eval()

	def classify(self, image):
		# Individua le regioni d'interesse (RoI)
		boxes = selective_search.selective_search(image, mode="fast")
		filtered_boxes = selective_search.box_filter(boxes, min_size=20, topN=80)

		# Per ogni regione ridimensionala e predici la classe con il modello
		fig, ax = plt.subplots(figsize=(6, 6))
		ax.imshow(image)
		classes = []
		for x1, y1, x2, y2 in filtered_boxes:
			box_img = image[x1:x2, y1:y2]
			if box_img.size > 0:
				bbox = mpatches.Rectangle((x1, y1), (x2 - x1), (y2 - y1), fill=False, edgecolor='red', linewidth=1)
				ax.add_patch(bbox)

				# Adatto il box al modello di CNN
				all_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((IMG_RESIZE_WIDTH, IMG_RESIZE_HEIGHT))])
				box_img = all_transforms(box_img)
				box_img = box_img.unsqueeze(0)

				# Ottengo la classe, faccio NMS e la salvo (solo se non è lo sfondo)
				output = self.model(box_img)
				_, predicted = torch.max(output.data, 1)
				predicted_class = OUTPUTS[predicted.item()]
				if predicted_class != "sfondo":
					# TODO: salvarsi in classes le box per fare NMS
					classes.append(predicted_class)
		plt.axis('off')
		plt.show()

		# Salva e restituisci i risultati
		return classes
