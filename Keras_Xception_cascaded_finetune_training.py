# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 15:02:12 2017

@author: Jason
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential,model_from_json
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint 
from keras.applications.xception import Xception
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Input

import scipy.io

import numpy as np
import numpy.matlib
import cv2
import random

import time

epoch=5
batch_size=8
Val_size  =10000
train_size=30000
image_list_path="ImageNet_list_all.txt"
fgsm_path      ="fgsm_noise3.txt"

image_list = [x[:] for x in open(image_list_path).readlines()]
random.shuffle(image_list)

Val_list=image_list[0:Val_size]
Train_list=image_list[Val_size:]

mapping= scipy.io.loadmat('Mapping.mat')

    
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
    #img = np.asarray(cv2.imread(img_path))[:, :, ::-1].astype('float32')
    img = central_crop(img, 0.875)
    img = cv2.resize(img, (299, 299))
    img = preprocess_input(img)
    img = img.reshape(-1, 299, 299, 3)
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
        return np.clip(image+ 0.15*np.sign(np.random.normal(size=input_shapes)),-1,1)
    elif r%6==1:
        i1=int(np.random.uniform(120,180)) 
        i2=int(np.random.uniform(120,180))
        r1=int(np.random.uniform(20,40))
        r2=int(np.random.uniform(20,40))
        image[0,i1:i1+r1,i2:i2+r2,:]=-1
        return np.clip(image,-1,1)
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
#    input_shapes=image.shape
    
    if   r%2==0:
#        return np.clip(image+ 0.125*np.sign(np.random.normal(size=input_shapes)),-1,1)
#    elif r%6==1:
#        return np.clip(image+ 0.08*np.sign(np.random.normal(size=input_shapes)),-1,1)
#    elif r%6==2:
#        return np.clip(image+ np.random.normal(0, 0.15, input_shapes),-1,1)
#    elif r%6==3:
        i=int(np.random.uniform(0, fgsm_noise.shape[0]))    
        s=int(np.random.uniform(0, 16-4+1+1))      # MAX_PERTURBATION will be always integer number between 4 and 16 inclusive.
        amp=np.arange(4,18)/16.
        return np.clip(image+ amp[s]*fgsm_noise[i],-1,1)
    else:
        i=int(np.random.uniform(0, fgsm_noise.shape[0]))
        return np.clip(image+ fgsm_noise[i],-1,1)
#    else:
#        return image #clean

def label_mapping(image_path):
    return int(mapping[image_path[49:58]][0])

def training_data_generation(Train_list, size, turns):
    print ('Preparing training data...')
    train_noisy = np.zeros((size, 299, 299, 3),dtype=np.float32)
    labels      = np.zeros((size, 1))
    valid=0
    for i in range(size):
        if i%1000==0:
            print ('%i images loaded' % i)
        img = np.asarray(cv2.imread(Train_list[(turns*size)+i][0:-1]))[:, :, ::-1]
        if len(img.shape)==3: # check for RGB image
            img = get_processed_image(img)        
            train_noisy[i] = train_add_noise(img,i)
            labels[i] = label_mapping(Train_list[(turns*size)+i][0:-1])
            valid=valid+1
    train_noisy=train_noisy[0:valid]
    labels=labels[0:valid]
        
    return train_noisy,labels
    
def validation_data_generation(Val_list, size):
    print ('Preparing validation data...')
    val_noisy = np.zeros((size, 299, 299, 3),dtype=np.float32)
    labels      = np.zeros((size, 1))
    valid=0
    for i in range(size):
        if i%1000==0:
            print ('%i images loaded' % i)
        img = np.asarray(cv2.imread(Val_list[i][0:-1]))[:, :, ::-1]
        if len(img.shape)==3: # check for RGB image
            img = get_processed_image(img)        
            val_noisy[i] = val_add_noise(img,i)
            labels[i] = label_mapping(Val_list[i][0:-1])
            valid=valid+1
    val_noisy=val_noisy[0:valid]
    labels=labels[0:valid]
        
    return val_noisy,labels

start_time = time.time()
#Only Used For visual comparison
with open('Denoise_ImageNet_fixed_dilation_finale.json','r') as f:
    De_model_original= model_from_json(f.read())
De_model_original.load_weights('Best_weights_ImageNet_fixed_dilation_finale.hdf5')


print ('De-noise model loading...')
with open('Denoise_ImageNet_fixed_dilation_finale.json','r') as f:
    De_model= model_from_json(f.read())
De_model.load_weights('Best_weights_ImageNet_fixed_dilation_finale.hdf5')

print ('Xception loading...')
classifier=Xception()

#classifier.trainable = False
#for layer in classifier.layers:
#    layer.trainable = False


'''
cas_classifier = Sequential()
cas_classifier.add(De_model)
cas_classifier.add(classifier)
'''

x = Input(shape=((299,299,3)))

denoise_result=De_model(x)
class_prediction=classifier(denoise_result)

cas_classifier = Model(inputs=x, outputs=class_prediction)

sgd = SGD(lr=0.000025, decay=5*1e-8, momentum=0.9, nesterov=True)
cas_classifier.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

with open('Finetuned_DXception_finale_7.json','w') as f:    # save the model
    f.write(cas_classifier.to_json())
    
checkpointer = ModelCheckpoint(filepath='Finetuned_DXception_finale_7.hdf5', verbose=1, save_best_only=False, mode='min')  #saves the model weights after each epoch if the validation loss decreased

(val_noisy, val_labels)=validation_data_generation(Val_list, Val_size)
val_labels=to_categorical(val_labels, num_classes=1000)

NPredicted_Labels = np.argmax(classifier.predict(val_noisy),1)
Noisy_accuracy = np.mean(np.equal(NPredicted_Labels,np.argmax(val_labels, axis=1)))
print('Noisy accuracy on Xception before fine-tunning:' + str(Noisy_accuracy))
         
#val_denoised=De_model.predict(val_noisy, verbose=1,batch_size=batch_size)   
NPredicted_Labels = np.argmax(cas_classifier.predict(val_noisy),1)
Noisy_accuracy = np.mean(np.equal(NPredicted_Labels,np.argmax(val_labels, axis=1)))
print('Noisy accuracy on cascaded_Xception before fine-tunning:' + str(Noisy_accuracy))


for i in range(5):
    
    print ('%.i-th Training data loading...' %(i+1))
    
    print ('data loading...')
    (train_noisy, train_labels)=training_data_generation(Train_list, train_size,i)    
    train_labels=to_categorical(train_labels, num_classes=1000)
    
    #train_denoised=De_model.predict(train_noisy, verbose=1,batch_size=batch_size) 
    
    print 'training...'    
    hist=cas_classifier.fit(train_noisy, train_labels, epochs=epoch, batch_size=batch_size, verbose=1, shuffle=True, validation_data=(val_noisy,val_labels), callbacks=[checkpointer])

    del train_noisy, train_labels#, train_denoised


print 'testing...'
NPredicted_Labels = np.argmax(classifier.predict(val_noisy),1)
Noisy_accuracy = np.mean(np.equal(NPredicted_Labels,np.argmax(val_labels, axis=1)))
print('Noisy accuracy on Xception after fine-tunning:' + str(Noisy_accuracy))

NPredicted_Labels = np.argmax(cas_classifier.predict(val_noisy),1)
Noisy_accuracy = np.mean(np.equal(NPredicted_Labels,np.argmax(val_labels, axis=1)))
print('Noisy accuracy on cascaded_Xception after fine-tunning:' + str(Noisy_accuracy))

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
plt.savefig('Learning_curve_cascaded_Xception_Finetune.png', dpi=150)

# show some image results
i=3
cv2.imwrite ('noisy_ImNet.bmp', ((np.squeeze(val_noisy[i])[:, :, ::-1]/2)+0.5)*255)

out=De_model.predict(val_noisy[0:40], verbose=1,batch_size=batch_size)
cv2.imwrite ('de_ImNet_finnetuned.bmp', np.squeeze(((out[i][:, :, ::-1]/2)+0.5)*255))

out=De_model_original.predict(val_noisy[0:40], verbose=1,batch_size=batch_size)
cv2.imwrite ('de_ImNet_original.bmp', np.squeeze(((out[i][:, :, ::-1]/2)+0.5)*255))

plot_model(classifier, to_file='model.png',show_shapes=True)

end_time = time.time()
print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))



