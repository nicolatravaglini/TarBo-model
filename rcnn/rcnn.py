from constants import *
import torch
import selective_search
from torchvision import transforms


class RCNN:
    def __init__(self):
        self.model = torch.load("../model/model.pt").eval()

    def classify(self, image):
        # Individua le regioni d'interesse (RoI)
        boxes = selective_search.selective_search(image, mode="fast")
        filtered_boxes = selective_search.box_filter(boxes, min_size=20, topN=80)

        # Per ogni regione ridimensionala e predici la classe con il modello
        classes = []
        for x1, y1, x2, y2 in filtered_boxes:
            box_img = image[x1:x2-x1, y1:y2-y1]
            resizer = transforms.Resize(IMG_RESIZE_WIDTH, IMG_RESIZE_HEIGHT)
            resized_img = resizer.forward(box_img)
            output = self.model(resized_img)
            _, predicted = torch.max(output.data, 1)
            classes.append(predicted)

        # Salva e restituisci i risultati
        return classes