from constants import *
from PIL import Image
import numpy
from rcnn.rcnn import RCNN
from flask import Flask, request

app = Flask(__name__)


@app.route("/", methods=["POST"])
def main():
    file = request.files["immagine"]
    file.save(file.filename)

    img = numpy.array(Image.open("test/IMG20230531164636.jpg").resize((IMG_ORIGINAL_WIDTH, IMG_ORIGINAL_HEIGHT)))
    rcnn = RCNN()
    classes = rcnn.classify(img)

    return classes
