#!/usr/bin/env python
# coding: utf-8

# In[85]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[88]:


dir='E:/my data/fruits'

categories=['Banana', 'Sweetcorn']

data=[]

for category in categories:
    path=os.path.join(dir, category)
    label=categories.index(category)
    for img in os.listdir(path):
        imgpath=os.path.join(path,img)
        fruit_img= cv2.imread(imgpath,0)
        try:
            fruit_img= cv2.resize(fruit_img,(300,300))
            image=np.array(fruit_img).flatten()
            data.append([image,label])
        except exception as e:
            pass


# In[89]:


pick_in=open('E:/my data/petimage/data1.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()


# In[90]:


pick_in=open('E:/my data/petimage/data1.pickle', 'rb')
data=pickle.load(pick_in)
pick_in.close()


# In[94]:


random.shuffle(data)
features=[]
labels=[]

for feature, label in data:
    features.append(feature)
    labels.append(label)
    
xtrain, xtest, ytrain, ytest=train_test_split(features,labels, test_size=0.20)

model=SVC(C=1, kernel='poly', gamma='auto')
model.fit(xtrain, ytrain)

pick=open('E:/my data/fruits/fruitmodel.sav', 'wb')
pickle.dump(model,pick)
pick.close()

pick=open('E:/my data/fruits/fruitmodel.sav', 'rb')
model=pickle.load(pick)
pick.close()

prediction=model.predict(xtest)

accuracy=model.score(xtest, ytest)

print('Accuracy', accuracy)

print('Prediction is ', categories[prediction[0]])

myfruit=xtest[7].reshape(300,300)

plt.imshow(myfruit, cmap='gray')

plt.show()


# In[ ]:




