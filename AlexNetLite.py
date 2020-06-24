# -*- coding: utf-8 -*-
# Stop from disconnecting

from google.colab import drive
drive.mount('/content/gdrive')

"""# Importing Necessary Libraries"""

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import tensorflow.keras 

from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras import backend as K 
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model 
import numpy as np 
from tensorflow.keras.preprocessing import image 

import matplotlib.pyplot as plt 
# %matplotlib inline

"""# Hyper-Parameters Initialization"""

epochs = 90
batch_size = 100
training_samples = 4000
validation_samples = 600
img_width = 227
img_height = 227
channels = 3
input_shape = (img_width,img_height,channels)

"""# Data Generator"""

train_data_dir = 'Small_set_cats_vs_dogs/train'
validation_data_dir = 'Small_set_cats_vs_dogs/val'

train_datagen = ImageDataGenerator( 
    rescale=1. / 255,
    shear_range=0.1, 
    zoom_range=0.2, 
    rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
    )

# Validation Data
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Original Data
train_generator = train_datagen.flow_from_directory( 
    train_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size
    )

# Validation Data
validation_generator = val_datagen.flow_from_directory( 
    validation_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size
    )

"""# Image Augmentation"""

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()    
  
augmented_images = [train_generator[0][0][0] for i in range(10)]
plotImages(augmented_images)

"""# CNN Model"""

model = Sequential()
model.add(Conv2D(filters=96, 
                 kernel_size=(11, 11), 
                 strides=(4,4),
                 input_shape=input_shape, 
                 activation='relu')) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3),
                       strides=(2, 2))) 

model.add(Conv2D(filters=256, 
                 kernel_size=(5, 5),
                 padding='same',
                 activation='relu')) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3),
                       strides=(2, 2))) 

model.add(Conv2D(filters=384, 
                 kernel_size=(3, 3), 
                 padding='same',
                 activation='relu')) 

model.add(Conv2D(filters=384, 
                 kernel_size=(3, 3), 
                 padding='same',
                 activation='relu')) 

model.add(Conv2D(filters=256, 
                 kernel_size=(3, 3), 
                 padding='same',
                 activation='relu')) 
model.add(MaxPooling2D(pool_size=(3, 3),
                       strides=(2, 2))) 

model.add(Flatten()) 
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy'])

"""# Model Layout and Visualization"""

model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

"""# Callbacks [Bonus]"""

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=7, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

snapshot_name = 'dummy_model_'
checkpoint = ModelCheckpoint(filepath=snapshot_name+".{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

"""# Training and Validation"""

# Training on data
history = model.fit_generator(train_generator, 
    steps_per_epoch = training_samples // batch_size, 
    epochs = epochs, validation_data = validation_generator, 
    validation_steps = validation_samples // batch_size,
    callbacks = [reduce_lr]
    )

model.save_weights('model_saved.h5')

"""# Evaluating the test data"""

test_data_dir = 'Small_set_cats_vs_dogs/test'

test_datagen = ImageDataGenerator(rescale=1. / 255) 

test_generator = test_datagen.flow_from_directory( 
    test_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size
    )

model.evaluate_generator(test_generator, verbose=2)

"""# Plotting Accuracy and Loss Graph"""

# Ploting on data
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('Acc.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('Loss.png')
plt.show()

"""# Predicting some sample images"""

img = image.load_img('/content/Small_set_cats_vs_dogs/test/cat/cat.6059.jpg', target_size=(img_width, img_height))
img_2 = image.load_img('/content/Small_set_cats_vs_dogs/test/dog/dog.5026.jpg', target_size=(img_width, img_height))
plt.imshow(img)
plt.show()
plt.imshow(img_2)
plt.show()

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
y = image.img_to_array(img_2)
y = np.expand_dims(y, axis=0)
images = np.vstack([x,y])

# Results from data

# Prediction result
pred = model.predict(images)
print (pred)
print()

# Class Prediction
pred_class = model.predict_classes(images)
print (pred_class)