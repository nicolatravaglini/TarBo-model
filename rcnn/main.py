from constants import *
from PIL import Image
import numpy
from rcnn import RCNN


def main():
	img = numpy.array(Image.open("test/IMG20230527190802.jpg").resize((IMG_ORIGINAL_WIDTH, IMG_ORIGINAL_HEIGHT)))
	rcnn = RCNN()
	classes = rcnn.classify(img)
	print(classes)


if __name__ == "__main__":
	main()
