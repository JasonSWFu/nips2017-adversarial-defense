# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 13:40:05 2017

@author: Jason
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.local import LocallyConnected2D
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import scipy.io

import numpy as np
import numpy.matlib
import cv2
import random

import time

epoch=1
batch_size=8
Val_size=40
train_size=350
image_list_path="ImageNet_list_all.txt"
#Val_list_path  ="ImageNet_val_list_all.txt"
fgsm_path      ="fgsm_noise3.txt"

image_list = [x[:] for x in open(image_list_path).readlines()]
random.shuffle(image_list)

Val_list=image_list[0:Val_size]
Train_list=image_list[Val_size:]
     
# This function comes from Google's ImageNet Preprocessing Script
def central_crop(image, central_fraction):
    """Crop the central region of the image.
    Remove the outer parts of an image but retain the central region of the
    image along each dimension. If we specify central_fraction = 0.5, this
    function returns the region marked with "X" in the below diagram.
       --------
      |        |
      |  XXXX  |
      |  XXXX  |
      |        |   where "X" is the central 50% of the image.
       --------
    Args:
    image: 3-D array of shape [height, width, depth]
    central_fraction: float (0, 1], fraction of size to crop
    Raises:
    ValueError: if central_crop_fraction is not within (0, 1].
    Returns:
    3-D array
    """
    if central_fraction <= 0.0 or central_fraction > 1.0:
        raise ValueError('central_fraction must be within (0, 1]')
    if central_fraction == 1.0:
        return image

    img_shape = image.shape
    # depth = img_shape[2]
    fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
    bbox_h_start = int(np.divide(img_shape[0], fraction_offset))
    bbox_w_start = int(np.divide(img_shape[1], fraction_offset))

    bbox_h_size = int(img_shape[0] - bbox_h_start * 2)
    bbox_w_size = int(img_shape[1] - bbox_w_start * 2)

    image = image[bbox_h_start:bbox_h_start+bbox_h_size,
                  bbox_w_start:bbox_w_start+bbox_w_size]
    return image

def preprocess_input(img):
    img = np.divide(img, 255.0)
    img = np.subtract(img, 0.5)
    img = np.multiply(img, 2.0)
    return img                  # -1 to +1
    
def get_processed_image(img):
    # Load image and convert from BGR to RGB
    #img = np.asarray(cv2.imread(img_path))[:, :, ::-1]
    img = central_crop(img, 0.875)
    img = cv2.resize(img, (299, 299))
    img = preprocess_input(img)
    #img = img.reshape(-1, 299, 299, 3)
    return img
    
    
def load_fgsm_noise(list_path, eps):
    print ('Preparing fgsm noise...')
    file_list = [x[:] for x in open(list_path).readlines()]
    random.shuffle(file_list)
    size=len(file_list)
    fgsm_noise = np.zeros((size, 299, 299, 3),dtype=np.float32)
    for i in range(size):
        img = np.asarray(cv2.imread(file_list[i][0:-1]),dtype=np.float32)[:, :, ::-1]
        img = (img-eps)/255.0*2      
        fgsm_noise[i] = img        
    return fgsm_noise
    
fgsm_noise=load_fgsm_noise(fgsm_path, 16)


def train_add_noise(image,r):
    input_shapes=image.shape
    
    if   r%6==0:
        return np.clip(image+ 0.14*np.sign(np.random.normal(size=input_shapes)),-1,1)
    elif r%6==1:
        return np.clip(image+ 0.1*np.sign(np.random.normal(size=input_shapes)),-1,1)
    elif r%6==2:
        return np.clip(image+ np.random.normal(0, 0.18, input_shapes),-1,1)
    elif r%6==3:
        i=int(np.random.uniform(0, fgsm_noise.shape[0]))    
        s=int(np.random.uniform(0, 16-4+1+1))      # MAX_PERTURBATION will be always integer number between 4 and 16 inclusive.
        amp=np.arange(4,18)/16.
        return np.clip(image+ amp[s]*fgsm_noise[i],-1,1)
    elif r%6==4:
        i=int(np.random.uniform(0, fgsm_noise.shape[0]))
        return np.clip(image+ fgsm_noise[i],-1,1)
    else:
        return image #clean

def val_add_noise(image,r):
    input_shapes=image.shape
    
    if   r%6==0:
        return np.clip(image+ 0.125*np.sign(np.random.normal(size=input_shapes)),-1,1)
    elif r%6==1:
        return np.clip(image+ 0.08*np.sign(np.random.normal(size=input_shapes)),-1,1)
    elif r%6==2:
        return np.clip(image+ np.random.normal(0, 0.15, input_shapes),-1,1)
    elif r%6==3:
        i=int(np.random.uniform(0, fgsm_noise.shape[0]))    
        s=int(np.random.uniform(0, 16-4+1+1))      # MAX_PERTURBATION will be always integer number between 4 and 16 inclusive.
        amp=np.arange(4,18)/16.
        return np.clip(image+ amp[s]*fgsm_noise[i],-1,1)
    elif  r%6==4:
        i=int(np.random.uniform(0, fgsm_noise.shape[0]))
        return np.clip(image+ fgsm_noise[i],-1,1)
    else:
        return image #clean
 

def training_data_generation(Train_list, size, turns):  
    print ('Preparing training data...')
    train_clean = np.zeros((size, 299, 299, 3),dtype=np.float32)
    train_noisy = np.zeros((size, 299, 299, 3),dtype=np.float32)
    for i in range(size):
        if i%1000==0:
            print ('%i images loaded' % i)
        img = np.asarray(cv2.imread(Train_list[(turns*size)+i][0:-1]))[:, :, ::-1]
        if len(img.shape)==3: # check for RGB image
            img = get_processed_image(img)        
            train_noisy[i] = train_add_noise(img,i)
            train_clean[i] = img
        
    return train_noisy, train_clean    

def validation_data_generation(Val_list, size):  
    print ('Preparing validation data...')
    val_clean = np.zeros((size, 299, 299, 3),dtype=np.float32)
    val_noisy = np.zeros((size, 299, 299, 3),dtype=np.float32)
    for i in range(size):
        img = np.asarray(cv2.imread(Val_list[i][0:-1]))[:, :, ::-1]
        if len(img.shape)==3: # check for RGB image
            img = get_processed_image(img)        
            val_noisy[i] = val_add_noise(img,i)
            val_clean[i] = img
        
    return val_noisy, val_clean    


start_time = time.time()

print 'model building...'
model = Sequential()

model.add(Conv2D(32, (9, 9), dilation_rate=(3, 3), padding='same', data_format='channels_last', input_shape=(299,299,3)))
model.add(BatchNormalization(axis=-1))
model.add(ELU())
model.add(Dropout(0.07))

model.add(Conv2D(48, (7, 7), dilation_rate=(4, 4), padding='same',  data_format='channels_last'))
model.add(BatchNormalization(axis=-1))
model.add(ELU())
model.add(Dropout(0.07))

model.add(Conv2D(48, (5, 5), dilation_rate=(4, 4), padding='same',  data_format='channels_last'))
model.add(BatchNormalization(axis=-1))
model.add(ELU())
model.add(Dropout(0.07))

model.add(Conv2D(32, (5, 5), dilation_rate=(4, 4), padding='same',  data_format='channels_last'))
model.add(BatchNormalization(axis=-1))
model.add(ELU())
model.add(Dropout(0.07))

model.add(Conv2D(16, (3, 3), dilation_rate=(4, 4), padding='same',  data_format='channels_last'))
model.add(BatchNormalization(axis=-1))
model.add(ELU())
model.add(Dropout(0.07))

model.add(Conv2D(3, (3, 3),  padding='same', data_format='channels_last'))
model.add(Activation('tanh'))

model.compile(loss='mse', optimizer="Nadam")

with open('Denoise_ImageNet_fixed_dilation_test.json','w') as f:    # save the model
    f.write(model.to_json())
checkpointer = ModelCheckpoint(filepath='Best_weights_ImageNet_fixed_dilation_test.hdf5', verbose=1, save_best_only=True, mode='min')  #saves the model weights after each epoch if the validation loss decreased

# Validation data generation
(val_noisy, val_clean)=validation_data_generation(Val_list, Val_size)

for i in range(7):

    print ('%.i-th Training data loading...' %(i+1))
    # Training data generation
    (train_noisy, train_clean)=training_data_generation(Train_list, train_size, i)

    print 'training...'    
    hist=model.fit(train_noisy, train_clean, epochs=epoch, batch_size=batch_size, verbose=1, shuffle=True, validation_data=(val_noisy,val_clean), callbacks=[checkpointer])
    
    out=model.predict(val_noisy[0:100], verbose=1,batch_size=batch_size)
    cv2.imwrite ('de_ImNet_'+str(i+1)+'.bmp', np.squeeze(((out[10][:, :, ::-1]/2)+0.5)*255))
    
    del train_noisy, train_clean

'''
print ('De-noise model loading...')
with open('Denoise_ImageNet_fixed_dilation2.json','r') as f:
    model= model_from_json(f.read())
model.load_weights('Best_weights_ImageNet_fixed_dilation2.hdf5')
    
'''


print 'testing...'
out=model.predict(val_noisy, verbose=1,batch_size=batch_size)

# Example of plotting the learning curve
TrainERR=hist.history['loss']
ValidERR=hist.history['val_loss']
print ('@%f, Minimun error:%f, at iteration: %i' % (hist.history['val_loss'][epoch-1], np.min(np.asarray(ValidERR)),np.argmin(np.asarray(ValidERR))+1))
print 'drawing the training process...'
plt.figure(1)
plt.plot(range(1,epoch+1),TrainERR,'b',label='TrainERR')
plt.plot(range(1,epoch+1),ValidERR,'r',label='ValidERR')
plt.xlim([1,epoch])
#plt.ylim([0.001,0.02])
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error')
plt.grid(True)
plt.show()
plt.savefig('Learning_curve_ImageNet_fixed_dilation.png', dpi=150)


## show some image results
i=10
cv2.imwrite ('clean_ImNet.bmp', np.squeeze(((val_clean[i][:, :, ::-1]/2)+0.5)*255))
cv2.imwrite ('noisy_ImNet.bmp', np.squeeze(((val_noisy[i][:, :, ::-1]/2)+0.5)*255))
cv2.imwrite ('de_ImNet.bmp', np.squeeze(((out[i][:, :, ::-1]/2)+0.5)*255))


end_time = time.time()
print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))




