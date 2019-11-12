# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data 
import pandas as pd
#import helper
from collections import OrderedDict
from PIL import Image
import seaborn as sns
import json
import pymysql
import matplotlib.pyplot as plt

import seaborn as sns

from image_classifier_utils import loading_model
from image_classifier_utils import process_image
from image_classifier_utils import imshow
from image_classifier_utils import predict
import json

import pymysql
# Serve model as a flask application

import pickle

from flask import Flask, request

model = None
app = Flask(__name__)



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
len (cat_to_name)




def load_model():
    global model
    # model variable refers to the global variable
    
    model = loading_model ('project_checkpoint.pth')
    print(model)


@app.route('/')
def home_endpoint():
    return 'run this command from a script or terminal - curl -X POST 0.0.0.0:5000/predict -H ''Content-Type: application/json'' '


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single samples
    if request.method == 'POST':
        
        
        
        
        #from pymysql.connector import Error
        
        
        mydb = pymysql.connect(
            host="localhost",
            user="root",
            passwd="",
            database="object_detection")
        
        print("connection successful")
        
        
        retreival_query="select image_url from images where serial_no = (select max(serial_no) from images)" 
        mycursor = mydb.cursor()
        mycursor.execute(retreival_query)
        myresult= mycursor.fetchall()
        
        #for i in myresult:
           # print(i)
        
        image_url_tuple = myresult[0]
        image_url_string = image_url_tuple[0]
        print (image_url_string)
        #file_path = request.get_json()  # Get data posted as a json
        #print(file_path)
        #data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        file_path=image_url_string#"/home/manojkhatokar/Downloads/udacity-image-classification-master/foreign_test_images/test_image5.jpeg"
        img = process_image(file_path)
        #img.shape
        imshow(img)
        probs, classes = predict (file_path, model, 5)



        #preparing class_names using mapping with cat_to_name
        
        class_names = [cat_to_name [item] for item in classes]
        
        #fig, (ax2) = plt.subplots(figsize=(6,9), ncols=2)
        plt.figure(figsize = (6,10))
        plt.subplot(2,1,2)
        #ax2.barh(class_names, probs)
        #ax2.set_aspect(0.1)
        #ax2.set_yticks(classes)
        #ax2.set_title('Flower Class Praobability')
        #ax2.set_xlim(0, 1.1)     
        
        sns.barplot(x=probs, y=class_names, color= 'green');
        
        #width = 1/5
        #plt.subplot(2,1,2)
        #plt.bar (classes, probs, width, color = 'blue')
        plt.show()
        
        #print (probs)
        #print (classes)
        result=class_names[0]
        print(result)
        return result
  # runs globally loaded model on the data
        #return str(prediction[0])


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=5000)
    #get_prediction()
