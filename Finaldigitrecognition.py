#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from PIL import Image,ImageOps

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

#Mnist dataset
mnist = fetch_openml('mnist_784', version=1)
X,y = mnist['data'], mnist["target"]


# In[44]:


class DigitRecognizer():
    def __init__(self, X,y):
        #SplitX X,y Data, each row of X will be shape 28x28 (3 dimensional matrix), y will be 1d array
        self.X_train, self.X_test, self.y_train, self.y_test = X[:15000], X[15000:20000], y[:15000], y[15000:20000] 
         # Need to reshape training data into (#of training examples) x 784
        self.X_train = self.X_train.reshape(len(self.y_train), 784)
        # Need to reshape X test data into (#of test examples) x 784
        self.X_test = self.X_test.reshape(len(self.y_test), 784) 
        # Neural Network classifier
        self.model = OneVsRestClassifier(SVC())
        self.model_fitted = False
        self.predictions = None
        self.train_predictions = None


    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)
        self.model_fitted  = True
        self.predictions = self.model.predict(self.X_test)
        self.train_predictions = self.model.predict(self.X_train);
      
    def predict_new(self, X_new):
        X_new = X_new.reshape(1,784)
        prediction = self.model.predict(X_new)
        return prediction

    def show_train_metrics(self):  #Metrics for training data
        # Generally, the closer these numbers are to 1, the better. Should each be at least 80%
        accuracy = accuracy_score(self.y_train, self.train_predictions)
        precision = precision_score(self.y_train, self.train_predictions)
        recall = precision_score(self.y_train, self.train_predictions)
        print(f"Precision Train: {precision}")
        print(f"Recall Train: {recall}")
        print(f"Accuracy Train: {accuracy}")
    def show_test_metrics(self): #Metrics for test data
        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions)
        recall = precision_score(self.y_test, self.predictions)
        print(f"Precision Test: {precision}")
        print(f"Recall Test: {recall}")
        print(f"Accuracy Test: {accuracy}")
    def show_shapes(self):
        print(self.y_train.shape)
        print(self.train_predictions.shape)


# In[ ]:





# In[45]:


# Execute Model
classifier = DigitRecognizer(X,y)  #Input X,y from beginning of the code (from mnist)


# In[ ]:


classifier.fit_model()             #Also makes predictions, saves in self.predictions


# In[ ]:


classifier.show_shapes()


# In[ ]:


accuracy = accuracy_score(classifier.y_train, classifier.train_predictions)
precision = precision_score(classifier.y_train, classifier.train_predictions, average="macro")
recall = recall_score(classifier.y_train, classifier.train_predictions, average="macro")
print(f"Accuracy Train: {accuracy}")
print(f"Recall Train: {recall}")
print(f"Precision Train: {precision}")


# In[ ]:


accuracy = accuracy_score(classifier.y_test, classifier.predictions)
precision = precision_score(classifier.y_test, classifier.predictions, average = "macro")
recall = precision_score(classifier.y_test, classifier.predictions, average = "macro")
print(f"Precision Test: {precision}")
print(f"Recall Test: {recall}")
print(f"Accuracy Test: {accuracy}")


# In[ ]:


import joblib
joblib.dump(classifier, "digimodel.pkl")


# In[ ]:


print("Past first area")
y_pred = classifier.predictions
print(y_pred)
classifier.show_train_metrics()    # Show the evaluation metrics (accuracy, precision, recall)
#classifier.show_test_metrics()     


# Note: Finish optimizing the model on both the training and test data before making predictions on completely new, hand-drawn images
# Get New Image, predict using ML Model 
# Using PIL module to open, convert image to 28 x 28, make it grayscale (instead of rgb), convert to array, plot image
img = Image.open('temp.png')
img_small = img.resize((28,28))
img_gray = ImageOps.grayscale(img_small)
# Invert the image because, for some reason, the background becomes all black with white number
img_gray = ImageOps.invert(img_gray)  
img_gray_arr = np.array(img_gray)
plt.imshow(img_gray_arr, cmap='binary')   # Plotting
plt.show()
print(img_gray_arr.shape)        # Shape will be 28x28

X_new = img_gray_arr
new_pred = classifier.predict_new(X_new)
print(new_pred)


# In[ ]:


modelloaded = joblib.load("digimodel.pkl")
for k in range(1, 10):
    img = Image.open(f'temp{k}.png')
    img_small = img.resize((28,28))
    img_gray = ImageOps.grayscale(img_small)
    # Invert the image because, for some reason, the background becomes all black with white number
    img_gray = ImageOps.invert(img_gray)  
    img_gray_arr = np.array(img_gray)
    plt.imshow(img_gray_arr, cmap='binary')  
    print(modelloaded.predict_new(img_gray_arr))
    print(accuracy_score(classifier.y_test, classifier.predictions))

