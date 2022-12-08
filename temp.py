from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
from PIL import Image
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)



# Model saved with Keras model.save()
BONE_MODEL_PATH ='Multibones_mobilenet_acc85_val__90.h5'
LUNGS_MODEL_PATH ='multilungs_resnet152_fine_acc96_val91.h5'
EYES_MODEL_PATH ='eye_E20_ACC89_val96.h5'

# Load your trained model
bonemodel = load_model(BONE_MODEL_PATH)
lungsmodel = load_model(LUNGS_MODEL_PATH)
eyesmodel = load_model(EYES_MODEL_PATH)

#bone

def bone_model_predict(img_path, model):
    boneimg = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    bonex = image.img_to_array(boneimg)
    #bonex = np.true_divide(bonex, 255)
    ## Scaling
    bonex=bonex/255
    bonex = np.expand_dims(bonex, axis=0)
   
    bonepreds = model.predict(bonex)
    bonepreds=np.argmax(bonepreds, axis=1)
     
    if bonepreds==0:
        bonepreds="THE CLASS IS ELBOW AND IT IS NORMAL IMAGE"
    elif bonepreds==1:
        bonepreds="THE CLASS IS ELBOW AND IT IS ABNORMAL IMAGE"
    elif bonepreds==2:
        bonepreds="THE CLASS IS FINGER AND IT IS NORMAL IMAGE"
    elif bonepreds==3:
        bonepreds="THE CLASS IS FINGER AND IT IS ABNORMAL IMAGE"
    elif bonepreds==4:
        bonepreds="THE CLASS IS FOREARM AND IT IS NORMAL IMAGE"
    elif bonepreds==5:
        bonepreds="THE CLASS IS FOREARM AND IT IS ABNORMAL IMAGE"
    elif bonepreds==6:
        bonepreds="THE CLASS IS HANDS AND IT IS NORMAL IMAGE"  
    elif bonepreds==7:
        bonepreds="THE CLASS IS HANDS AND IT IS ABNORMAL IMAGE"    
    elif bonepreds==8:
        bonepreds="THE CLASS IS HUMERUS AND IT IS NORMAL IMAGE" 
    elif bonepreds==9:
        bonepreds="THE CLASS IS HUMERUS AND IT IS ABNORMAL IMAGE"    
    elif bonepreds==10:
        bonepreds="THE CLASS IS SHOULDER AND IT IS NORMAL IMAGE" 
    elif bonepreds==11:
        bonepreds="THE CLASS IS SHOULDER AND IT IS ABNORMAL IMAGE"  
    elif bonepreds==12:
        bonepreds="THE CLASS IS WRIST AND IT IS NORMAL IMAGE"   
    elif bonepreds==13:
        bonepreds="THE CLASS IS WRIST AND IT IS ABNORMAL IMAGE"
    
    return bonepreds

#lungs

def lungs_model_predict(img_path, model):
    lungsimg = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    lungsx = image.img_to_array(lungsimg)
    #lungsx = np.true_divide(lungsx, 255)
    ## Scaling
    lungsx=lungsx/255
    lungsx = np.expand_dims(lungsx, axis=0)
   
    lungspreds = model.predict(lungsx)
    lungspreds=np.argmax(lungspreds, axis=1)
    if lungspreds==0:
        lungspreds="Bengin"
    elif lungspreds==1:
        lungspreds="COVID"
    elif lungspreds==2:
        lungspreds="Malignant"
    elif lungspreds==3:
        lungspreds="Normal"
    elif lungspreds==4:
        lungspreds="Pneumonia"
    else:
        lungspreds="TB"

    return lungspreds

#eyes

def eyes_model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    #x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="CNV"
    elif preds==1:
        preds="DME"
    elif preds==2:
        preds="DRUSEN"    
    else:
        preds="normal"
    
    return preds

@app.route('/', methods=['GET', 'POST'])
def index():
    # Main page
    return render_template('index.html')

# bone

@app.route('/bonepredict', methods=['GET', 'POST'])
def boneupload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        bonepreds = bone_model_predict(file_path, bonemodel)
        boneresult= bonepreds
        return boneresult
    return None

# lungs

@app.route('/lungspredict', methods=['GET', 'POST'])
def lungsupload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        lungspreds = lungs_model_predict(file_path, lungsmodel)
        lungsresult=lungspreds
        return lungsresult
    return None

# eyes

@app.route('/eyespredict', methods=['GET', 'POST'])
def eyesupload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = eyes_model_predict(file_path, eyesmodel)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)



