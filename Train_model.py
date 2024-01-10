#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import nibabel as nib
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from nilearn.image import resample_img
import numpy as np

from scipy import ndimage
import shutil

# In[4]:


from nilearn import image as nli
from nilearn import plotting
from nilearn.plotting import plot_roi

from volumentations import *

# In[17]:

os.environ["CUDA_VISIBLE_DEVICES"]="1"


#from tensorflow.compat.v1.keras.backend import set_session
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#set_session(tf.compat.v1.Session(config=config))


# Delete the folders before starting, to not double save images 
shutil.rmtree('data_ad')
shutil.rmtree('data_cn')

tf.random.set_seed(42)

from zipfile import ZipFile

# - change to the appropriate directory 
with ZipFile("/raid/DATASETS/selim_datasets/data/Mariana_Filipa/LMCI_final.zip", 'r') as file:
   # Extract all the contents of zip file in current directory
   listOfFileNames = file.namelist()
   for fileName in listOfFileNames:
      if fileName.endswith('SS.img') or fileName.endswith('SS.hdr'):
        file.extract(fileName, "data_ad")
   file.close()

with ZipFile("/raid/DATASETS/selim_datasets/data/Mariana_Filipa/EMCI_final_.zip", 'r') as file:
   # Extract all the contents of zip file in current directory
   listOfFileNames = file.namelist()
   for fileName in listOfFileNames:
      if fileName.endswith('SS.img') or fileName.endswith('SS.hdr'):
        file.extract(fileName, "data_cn")
   file.close()


# In[18]:


ad=[]
cn=[]
# - create the arrays with ad and cn images 
for subdir, dirs, files in os.walk("data_ad"):
    for fil in files:
        filepath = subdir + os.sep + fil
        if filepath.endswith("SS.img"):
            ad.append(filepath)
for subdir, dirs, files in os.walk("data_cn"):
    for fil in files:
        filepath = subdir + os.sep + fil
        if filepath.endswith("SS.img"):
            cn.append(filepath)

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    #scan = resample_img(scan, target_affine=np.eye(3)*4., interpolation='nearest')
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    #min = -1000
    #max = 400
    #volume[volume < min] = min
    #volume[volume > max] = max
    #volume = (volume - min) / (max - min)
    #volume = volume.astype("float32")
    
    min = np.min(volume)
    max = np.max(volume)

    if max > min:
        img_norm = (volume - min) / (max - min)
    else:
        img_norm = volume * 0.
    img_norm = img_norm.astype("float32")
    return img_norm


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 96
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[0]
    current_width = img.shape[1]
    current_height = img.shape[2]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    #img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    volume=volume[:,:,:,0]
    print(volume.shape)
    # Resize width, height and depth
    volume = resize_volume(volume)
    print(volume.dtype)
    
    return volume


# In[21]:


data_cn = np.array([process_scan(path) for path in cn])
label_cn = np.array([0 for path in cn])

data_ad = np.array([process_scan(path) for path in ad])
label_ad = np.array([1 for path in ad])

data = np.concatenate((data_cn, data_ad), axis=0)
label = np.concatenate((label_cn, label_ad), axis=0)

print ("Dados de AD:", len(data_ad))
print ("Dados de CN:", len(data_cn))


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
X_trainnig, X_validation, y_trainnig, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print ("Dados de treino:", len(X_trainnig))
print ("Dados de validação:", len(X_validation))
print ("Dados de teste:", len(X_test))

# - to avoid repeating the process every time, save an array with all the information already processed
#np.save("/raid/DATASETS/selim_datasets/data/Models/train_image_emci", X_trainnig)
#np.save("/raid/DATASETS/selim_datasets/data/Models/train_label_emci", y_trainnig)
#np.save("/raid/DATASETS/selim_datasets/data/Models/validation_image_emci", X_validation)
#np.save("/raid/DATASETS/selim_datasets/data/Models/validation_label_emci", y_validation)
#np.save("/raid/DATASETS/selim_datasets/data/Models/teste_image_emci", X_test)
#np.save("/raid/DATASETS/selim_datasets/data/Models/teste_label_emci", y_test)

# - so all the lines of code above, can be replaced with np.load(array)

# In[28]:


train_loader = tf.data.Dataset.from_tensor_slices((X_trainnig, y_trainnig))
validation_loader = tf.data.Dataset.from_tensor_slices((X_validation, y_validation))


# In[29]:

import random

def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume

# - Implement various data augmentation functions 
def get_augmentation():
    return Compose([
        #Rotate((0, 0), (0, 0), (-15, 15), p=0.5),
        ElasticTransform((0, 0.25), interpolation=1, p=0.1),
        #Flip(0, p=0.5),
        #GaussianNoise(var_limit=(0, 1), p=0.2),
        #RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
    ], p=1.0)

def augmentor(img):
    aug = get_augmentation()
    data = {'image': img}
    aug_data = aug(**data)
    img     = aug_data['image']
    return np.ndarray.astype(img , np.float32)

def train_preprocessing(volume, label):
    """Process training data by applying the data augmentation technique choosen, adding a channel and convert to rgb code."""
    # Rotate volume
    #volume = rotate(volume)
    #volume = tf.image.flip_up_down(volume)
    volume = tf.numpy_function(augmentor, [volume], tf.float32)
    volume = tf.expand_dims(volume, axis=3)
    volume=tf.image.grayscale_to_rgb(volume)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel and convert to rgb."""
    volume = tf.expand_dims(volume, axis=3)
    volume=tf.image.grayscale_to_rgb(volume)
    return volume, label


# In[30]:

# - select the number of batches to use depending on the number of images 
batch_size = 3
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(X_trainnig))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(X_validation))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)


# In[25]:


import h5py
from tensorflow.keras import layers
from classification_models_3D.tfkeras import Classifiers
from keras.models import Model
from keras.layers import Dropout, Dense, Activation, GlobalAveragePooling3D


# In[32]:


#Different backbones available:
#### DenseNET
# - backbone='densenet121'
# - backbone='densenet169'
# - backbone='densenet201'
#### ResNET
# - backbone='resnet18'
# - backbone='resnet34'
# - backbone='resnet50'
# - backbone='resnet101'
# - backbone='resnet152'
#### ResNeXt 
# - backbone='resnext50'
# - backbone='resnext101'

#### SE-ResNET
# - backbone='seresnet18'
# - backbone='seresnet34'
# - backbone='seresnet50'
# - backbone='seresnet101'
# - backbone='seresnet152'
#### SE-ResNeXt 
# - backbone='seresnext50'
# - backbone='seresnext101'
#### VGG 
# - backbone='vgg16'
# - backbone='vgg19'
#### SE-Net 
# - backbone='senet154'
#### EfficientNet 
# - backbone='efficientnetb0'
# - backbone='efficientnetb1'
# - backbone='efficientnetb2'
# - backbone='efficientnetb3'
# - backbone='efficientnetb4'
# - backbone='efficientnetb5'
# - backbone='efficientnetb6'
# - backbone='efficientnetb7'

# - select the backbone and the input-shape
backbone='seresnet152'
Seresnet152 , preprocess_input = Classifiers.get(backbone)
model = Seresnet152(input_shape=(96, 128, 128, 3),include_top=False, weights='imagenet')

model.trainable = False

# - add a classification block 
y = model.layers[-1].output
y = GlobalAveragePooling3D()(y)
#y = Dense(512, activation='sigmoid', name='classification')(y)
y = Dropout(0.5)(y)
y = Dense(1, name='prediction')(y)
y = Activation('sigmoid')(y)
model_2 = Model(inputs=model.inputs, outputs=y)
#model_2.summary()


# In[33]:


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import Adam


# - select the learning rate, the number of epochs, optimizer and metrics
learning_rate = 0.0001
epochs = 20

optim = Adam(learning_rate=learning_rate)

loss_to_use = 'binary_crossentropy'
model_2.compile(optimizer=optim, loss=loss_to_use, metrics=keras.metrics.BinaryAccuracy())

cache_model_path = '/raid/DATASETS/selim_datasets/data/Models/'+'{}_temp.h5'.format(backbone)
best_model_path = '{}'.format(backbone) + '-{val_loss:.4f}-{epoch:02d}.h5'
# - implement some callbakcs to help the training process and avoid overfitting 
callbacks = [
    ModelCheckpoint(cache_model_path, monitor='val_loss', verbose=0, save_best_only=True),
    ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=15),
    ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=3, min_lr=1e-9, min_delta=1e-8, verbose=1, mode='min'),
    CSVLogger('history_{}_lr_{}.csv'.format(backbone, learning_rate), append=True)
]

model_2.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, verbose=1, max_queue_size=10, initial_epoch=0, callbacks=callbacks)


def unfreeze_model(model):
    # We unfreeze the top 80 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-80:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    #model.trainable = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=[keras.metrics.BinaryAccuracy()]
    )


unfreeze_model(model_2)


history_final = model_2.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    verbose=2,
    max_queue_size=10,
    initial_epoch=0,
    callbacks=callbacks
)


# In[73]:

# - save the model 
model_2.save('/raid/DATASETS/selim_datasets/data/Models/seresnet152_final_lmci___.h5')
file = pd.DataFrame(history_final.history)
file.to_csv('/raid/DATASETS/selim_datasets/data/Models/seresnet152_lmci___.csv')

