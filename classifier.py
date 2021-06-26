import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression 
from PIL import Image

X,y = fetch_openml("mnist_784",return_X_y = True,version = 1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=2500,train_size=7500)
X_train_scale = X_train/255
X_test_scale = X_test/255
clf = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(X_train_scale,y_train)

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resize = image_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resize,pixel_filter)
    image_bw_resize_invert_scale = np.clip(image_bw_resize-min_pixel,0,255)
    max_pixel = np.max(image_bw_resize)
    image_bw_resize_invert_scale = np.asarray(image_bw_resize_invert_scale)/max_pixel
    test_sample = np.array(image_bw_resize_invert_scale).reshape(1,784)
    test_prediction = clf.predict(test_sample)
    return test_prediction[0]