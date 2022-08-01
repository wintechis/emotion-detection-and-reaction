#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras

from keras.utils import np_utils
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers import BatchNormalization
import os

# Any results you write to the current directory are saved as output

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import make_classification
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# In[ ]:





# In[12]:


data_path = 'C:/Users/Datev Ma/Desktop/JupyterNotebook/CK+48/CK+48/'
classes = os.listdir(data_path) # 0 1 2 3

num_epoch=10

data_list =[]

def create_Data_List():
    for category in classes:
        category_path = os.path.join(data_path, category) 
        ImgsInCategory = os.listdir(category_path) # 0a.png 0b.png ...
#         print(ImgsInCategory[0])
        for img in ImgsInCategory:
            current_img=cv2.imread(os.path.join(category_path, img))
            #current_img=cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            current_img_resize=cv2.resize(current_img,(80,80))
            data_list.append(current_img_resize) #[ 24,  24,  24], [ 18,  18,  18], [ 10,  10,  10],...
            
create_Data_List()
        
def prepare_data(data_list):
    data_list = np.array(data_list) #keine kommas mehr
    data_list = data_list.astype('float32') 
    data_list = data_list/255 #[0.09411765 0.09411765 0.09411765]
#     print(data_list.shape) 
    return data_list
        
data_list = prepare_data(data_list)

# print(data_list)


# In[10]:


labels = []
for category in classes:
    category_path2 = os.path.join(data_path, category) 
    ImgsInCategory2 = os.listdir(category_path2)
    num_of_labels = len(ImgsInCategory2)
    labels.extend([int(category)] * num_of_labels) 
    
from collections import Counter
#print(Counter(labels))
labels = np.array(labels)    
# print(labels)

num_labels = 4

num_of_samples = len(data_list)
# labels = np.ones((num_of_samples,),dtype='int64')

# labels[0:135]=0 #135
# labels[135:210]=1 #54  75
# labels[210:417]=2 #177  207
# labels[417:501]=3 #75   84
# print(labels)

names = ['0','1','2','3']

def getLabel(id):
    return ['anger','fear','happy','sad'][id]


# In[4]:


print(data_list)
Y = np_utils.to_categorical(labels, num_labels)
# print(Y)
# print(data_list)
#Shuffle the dataset
x,y = shuffle(data_list,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
x_test=X_test
print(x)
print("______________________________________________________________________")
print(X_train)


# In[5]:


def create_model():
    input_shape=(80,80,3)

    model = Sequential()
    model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5), padding='same', activation = 'relu'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='RMSprop')
    
    return model


# In[6]:


model_custom = create_model()
model_custom.summary()


# In[7]:


from sklearn.model_selection import KFold


# In[8]:


kf = KFold(n_splits=5, shuffle=False) # 5 splits of test and training data


# In[9]:


from keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")


# In[10]:


batch_size = 8
EPOCHS = 200


# In[11]:


result = []
scores_loss = []
scores_acc = []
k_no = 0
for train_index, test_index in kf.split(x):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_Train_ = x[train_index]
    Y_Train = y[train_index]
    X_Test_ = x[test_index]
    Y_Test = y[test_index]
    print(X_Train_)

    #file_path = "/kaggle/working/weights_best_"+str(k_no)+".hdf5"
    #checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="loss", mode="min", patience=8)

    #callbacks_list = [checkpoint, early]

    model = create_model()
    #hist = model.fit_generator(aug.flow(X_Train_, Y_Train), epochs=EPOCHS,validation_data=(X_Test_, Y_Test), callbacks=callbacks_list, verbose=0)
    hist = model.fit(X_Train_, Y_Train, batch_size=batch_size, epochs=EPOCHS, validation_data=(X_Test_, Y_Test), verbose=1)
    #model.load_weights(file_path)
    result.append(model.predict(X_Test_))
    score = model.evaluate(X_Test_,Y_Test, verbose=0)
    scores_loss.append(score[0])
    scores_acc.append(score[1])
    k_no+=1


# In[ ]:


model.save('model_8_50epoch80_CK48dataset.h5')


# In[ ]:


# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn import svm, datasets
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import ConfusionMatrixDisplay

# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# class_names = iris.target_names

# # Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# # Run classifier, using a model that is too regularized (C too low) to see
# # the impact on the results
# classifier = svm.SVC(kernel="linear", C=0.01).fit(X_train, y_train)

# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# titles_options = [
#     ("Confusion matrix, without normalization", None),
#     ("Normalized confusion matrix", "true"),
# ]
# for title, normalize in titles_options:
#     disp = ConfusionMatrixDisplay.from_estimator(
#         classifier,
#         X_test,
#         y_test,
#         display_labels=class_names,
#         cmap=plt.cm.Blues,
#         normalize=normalize,
#     )
#     disp.ax_.set_title(title)

#     print(title)
#     print(disp.confusion_matrix)

# plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




