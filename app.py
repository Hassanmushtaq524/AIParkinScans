from distutils.log import debug
import tensorflow as tf
import cv2 as cv2
import os
from pydoc import render_doc
from flask import Flask, render_template, request, redirect, url_for
import pickle
from skimage import feature
import matplotlib.pyplot as plt
import librosa
app = Flask(__name__)

spiralModel = pickle.load(open("spiralModel.pkl", "rb"))

UPLOAD_FOLDER = "./upload"
ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


prediction1 = 0
prediction2 = 0
# def preprocess(image):

def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

def allowed_file(filename):     
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/aips")
def aips():
    return render_template("aips.html")

@app.route("/resources")
def resources():
    return render_template("resources.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/spiral", methods=["GET", "POST"])
def spiral():
    if request.method == "POST":
        f = request.files["file"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
        f.save(path)
        # load the input image, convert it to grayscale, and resize
        # it to 200x200 pixels, ignoring aspect ratio
        image = cv2.imread(path)
        os.remove(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        # threshold the image such that the drawing appears as white
        # on a black background
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # quantify the image
        features = quantify_image(image)

        prediction = spiralModel["classifier"].predict(features.reshape(-1, 12996))
   
    if prediction == 0:
        return redirect("/healthy")
    elif prediction == 1:
        return redirect("/parkinsons")




@app.route("/parkinsons")
def parkinsons():
    return render_template("haveparkinsons.html")


@app.route("/healthy")
def healthy():
    return render_template("healthy.html")

if __name__ == "__main__":
    app.run(debug=True)
