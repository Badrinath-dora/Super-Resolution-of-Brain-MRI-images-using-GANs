from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import io
import os
import cv2
import keras
from PIL import Image
import numpy as np
import base64
import matplotlib.pyplot as plt
from PIL import Image
from ipywidgets import FileUpload
from IPython.display import display
from keras.models import load_model
from numpy.random import randint
import cv2
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__,static_folder='static')
def read_image(image1):
    image= image1  
    resized_image = image.np.clerresize(*(32, 32), Image.ANTIALIAS)
    resized_image.save("resized_image.jpg")
    generator = load_model('gen_e_20.h5', compile=False)
    [X1, X2] = generator.predict(resized_image)
    ix = randint(0, len(X1), 1)
    src_image, tar_image = X1[ix], X2[ix]
    gen_image = generator.predict(src_image)
    plt.figure(figsize=(16, 8))
    plt.subplot(232)
    plt.title('Superresolution');
    plt.imshow(gen_image[0,:,:,:])
    return gen_image


@app.route('/',methods=['GET','POST'])
def home():
     return render_template('index.html')

@app.route('/index',methods=['GET','POST'])

def index():
    if request.method == 'POST':
        image1 = request.files['file'].read()
        image1 = cv2.imdecode(np.frombuffer(image1, np.uint8), cv2.IMREAD_COLOR)
        y_pred=read_image(image1)
        input_buffer = io.BytesIO() 
        output_buffer =io.BytesIO()
        output_pil_image = Image.fromarray(y_pred)
        output_pil_image.save(output_buffer, format='JPG')       
        mask_image = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        return render_template('result.html', mask_image=mask_image)
    else:
        return render_template('result.html')
if __name__ == '_main_':
    app.run(debug=True)