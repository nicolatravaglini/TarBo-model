from ..constants import *
from PIL import Image
import numpy
from ..rcnn.rcnn import RCNN
from flask import Flask, request, jsonify
import os

app = Flask(__name__)


@app.route("/", methods=["POST"])
def main():
    # Salvo l'immagine ricevuta
    file = request.files["immagine"]
    file.save(file.filename)

    # Apro l'immagine e ne trovo le classificazioni
    img = numpy.array(Image.open(file.filename).resize((IMG_ORIGINAL_WIDTH, IMG_ORIGINAL_HEIGHT)))
    rcnn = RCNN()
    classes = rcnn.classify(img)

    # Elimino l'immagine
    os.remove(file.filename)

    # Restituisco i risultati
    return jsonify(
        classes=classes
    )
