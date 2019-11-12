
    # Imports here

import matplotlib.pyplot as plt

import seaborn as sns

from image_classifier_utils import loading_model
from image_classifier_utils import process_image
from image_classifier_utils import imshow
from image_classifier_utils import predict
import json

import pymysql
#from pymysql.connector import Error

##################################################################################################################################################
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






###############################################################################################################################################


















with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
len (cat_to_name)




model_verify = loading_model ('project_checkpoint.pth')
model_verify

image_path = image_url_string#'/home/manojkhatokar/Downloads/udacity-image-classification-master/foreign_test_images/test_image5.jpeg'
img = process_image(image_path)
#img.shape
imshow(img)


model = model_verify #using the restored one 
file_path = image_url_string#'/home/manojkhatokar/Downloads/udacity-image-classification-master/foreign_test_images/test_image5.jpeg' #an example from test set

img = process_image (file_path)
imshow (img)
plt.show()
probs, classes = predict (file_path, model, 5)



#preparing class_names using mapping with cat_to_name

class_names = [cat_to_name [item] for item in classes]

#fig, (ax2) = plt.subplots(figsize=(6,9), ncols=2)
plt.figure(figsize = (6,10))
plt.subplot(2,1,2)
#ax2.barh(class_names, probs)
#ax2.set_aspect(0.1)
#ax2.set_yticks(classes)
#ax2.set_title('Flower Class Probability')
#ax2.set_xlim(0, 1.1)

sns.barplot(x=probs, y=class_names, color= 'green');

#width = 1/5
#plt.subplot(2,1,2)
#plt.bar (classes, probs, width, color = 'blue')
plt.show()

#print (probs)
#print (classes)
result=class_names[0]










#################################################################################################################################################################3





result_update_query= "insert into images(image) values (%s)"
mycursor.execute(result_update_query,result)
