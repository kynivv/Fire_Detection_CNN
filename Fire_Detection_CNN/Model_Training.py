import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization,Dense,SeparableConv2D,MaxPooling2D,Activation,Flatten,Dropout
from keras.callbacks import ModelCheckpoint

INIT_LR = 0.1
BATHC_SIZE = 64
NUM_EPOCHS = 50
lr_find = True

classes = ['Non_Fire', 'Fire']


images = []
labels = []
for c in classes:
    try:
        for img in os.listdir('Datasets/'+c):
            img = cv2.imread('Datasets/'+c+'/'+img)
            img = cv2.resize(img, (128,128))
            images.append(img)
            labels.append([0,1][c=='Fire'])
    except:
        pass

images = np.array(images,dtype='float32')/255.


labels = np.array(labels)
labels = np_utils.to_categorical(labels,num_classes=2)


d = {}

classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

d[0] = classWeight[0]
d[1] = classWeight[1]


X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.25, shuffle=True, random_state=42)


aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)


model = Sequential()

# CONV => RELU => POOL
model.add(SeparableConv2D(16,(7,7),padding='same',input_shape=(128,128,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# CONV => RELU => POOL
model.add(SeparableConv2D(32,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# CONV => RELU => CONV => RELU => POOL
model.add(SeparableConv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# first set of FC => RELU layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# second set of FC => RELU layers
model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# softmax classifier
model.add(Dense(len(classes)))
model.add(Activation("softmax"))

opt = SGD(learning_rate=INIT_LR, momentum=0.9, weight_decay=INIT_LR / NUM_EPOCHS)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath='output/checkpoint.h5',
                             monitor='val_accuracy',
                             save_best_only=True,
                             save_weights_only=False,
                             verbose=1)

H = model.fit(
    aug.flow(X_train, Y_train, batch_size= BATHC_SIZE),
    validation_data= (X_test,Y_test),
    steps_per_epoch= X_train.shape[0] // BATHC_SIZE,
    epochs= NUM_EPOCHS,
    class_weight=d,
    verbose=1,
    callbacks=[checkpoint]
)

model.save('output/fire_detection.h5')


N = np.arange(0,NUM_EPOCHS)

plt.figure(figsize=(12,8))

plt.subplot(121)
plt.title('Losses')
plt.plot(N, H.history['loss'], label='train_loss')
plt.plot(N, H.history['val_loss'], label='val_loss')

plt.subplot(122)
plt.title('Accuracy')
plt.plot(N, H.history['accuracy'], label='train_acc')
plt.plot(N, H.history['val_accuracy'], label='val_acc')

plt.legend()
plt.savefig('output/training_plot.png')