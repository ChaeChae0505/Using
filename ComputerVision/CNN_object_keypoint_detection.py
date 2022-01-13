import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras import backend as K
from keras import optimizers
import pandas as pd
from sklearn.utils import shuffle

#%% Config
batch_size = 64
nb_classes = 6
nb_epoch = 100
img_channel = 1
nb_filters = 32
nb_pool = 2
nb_conv = 3

#%% Read Csv
im_matrix = []
path = 'C:\\Users\\HYU\\Desktop\\800 images'
df = pd.read_csv('C:\\Users\\HYU\\Desktop\\lecture\\export_dataframe.csv',converters={"Price":int})
image_name = df.loc[:,'image_name']
image_name = image_name.tolist()
for i in range(np.shape(image_name)[0]):
    im = image_name[i]
    if type(im) == str:
        im1 = Image.open(path + '\\' + im)
        im2 = np.array(im1.convert('L'))
        im_matrix.append(im2)
X = np.expand_dims(im_matrix,axis=3)
Y = []
for i in range(np.shape(image_name)[0]):
    label = []
    if type(image_name[i]) == str:
        for j in range(6): # 6 is 2*number of keypoint
            y = int(df.loc[i][j])
            label.append(y/100) # 100 is size of image 
            # to convert from (0,100) to (0,1)
        Y.append(label)
Y = np.array(Y)
X = (X.astype('float64'))/255
Y = (Y.astype('float64'))
X, Y = shuffle(X, Y)

#%% Model creat
model = Sequential()

# input layer
model.add(BatchNormalization(input_shape=(100, 100, 1)))
model.add(Conv2D(24, (5, 5), kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))
# layer 2
model.add(Conv2D(36, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))
# layer 3
model.add(Conv2D(48, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))
# layer 4
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))
# layer 5
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Flatten())

#model.add(GlobalAveragePooling2D());
# layer 6
model.add(Dense(500, activation="relu"))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# layer 7
model.add(Dense(90, activation="relu"))
# layer 8
model.add(Dense(6)) # 6 is x,y values of 3 keypoint
#model.add(Activation('softmax'))
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
#%% Training

history=model.fit(X, Y, batch_size = 64, nb_epoch = 20,
          verbose = 1, validation_split=0.2) #0.2 is 20% of dataset for validation
#%%Save model
model.save("Trained_model_keypoint_1.h5")
#%%Load model
model = load_model("Trained_model_keypoint_1.h5")
#%%Test
a = '1_392'
path2 = 'C:\\Users\\HYU\\Desktop\\800 images'

listing = os.listdir(path2)
test_image = Image.open(path2 + '/' + a + '.jpg')
test_image = test_image.resize((100,100))
test_image_np = np.array(test_image.convert('L'))

test_image_np1 = np.expand_dims(test_image_np, axis = 0)
test_image_np2 = np.expand_dims(test_image_np1, axis = 3)
test_image_np2 = test_image_np2.astype('float32')
test_image_np2 /=255
predict = model.predict(test_image_np2)
predict1 = []
for i in range(6):
    if i%2 == 0:
        #print(predict1[0][j])
        predict1.append(int(predict[0][i]*100))
    else:
        predict1.append(int(predict[0][i]*100))
        #print(predict1[0][j])


x1_org = predict1[0]
y1_org = predict1[1]
x2_org = predict1[2]
y2_org = predict1[3]
x3_org = predict1[4]
y3_org = predict1[5]

plt.figure(figsize=(8, 8))
plt.imshow(test_image, cmap = 'gray')
plt.scatter(x1_org,y1_org, color = 'r')
plt.scatter(x2_org,y2_org, color = 'r')
plt.scatter(x3_org,y3_org, color = 'r')