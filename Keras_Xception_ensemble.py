# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:37:15 2017

@author: Jason
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from scipy.misc import imread

import tensorflow as tf


from keras.models import  model_from_json
from keras.applications.xception import Xception

import numpy as np





tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 8, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS

def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """

  
  images = np.zeros(batch_shape,dtype=np.float32)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = np.fliplr(imread(f, mode='RGB').astype('float32') / 255.0)  
    # Images for inception classifier are normalized to be in [-1, 1] interval.
      image=image * 2.0 - 1.0
    images[idx, :, :, :] = np.clip(image+ 0.04*np.sign(np.random.normal(size=image.shape)),-1,1)

    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape,dtype=np.float32)
      idx = 0
  if idx > 0:
    yield filenames, images
    




def main(_):
    
#    print ('De-noise model loading...')
#    with open('Denoise_ImageNet_fixed_dilation3.json','r') as f:
#        De_model= model_from_json(f.read())
#    De_model.load_weights('Best_weights_ImageNet_fixed_dilation3.hdf5')

    print ('classifier loading...')
    with open('Finetuned_DXception_finale_2.json','r') as f:
        classifier1= model_from_json(f.read())
    classifier1.load_weights('Finetuned_DXception_finale.hdf5')
    
    with open('Finetuned_DXception_finale_8.json','r') as f:
        classifier2= model_from_json(f.read())
    classifier2.load_weights('Finetuned_DXception_finale_8.hdf5')
    
    with open('Finetuned_DXception_finale_3.json','r') as f:
        classifier3= model_from_json(f.read())
    classifier3.load_weights('Finetuned_DXception_finale_3.hdf5')
    
    with open('Finetuned_DXception_finale_7.json','r') as f:
        classifier4= model_from_json(f.read())
    classifier4.load_weights('Finetuned_DXception_finale_7.hdf5')
          
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
 
    with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            print ('De-noising...')
            
            Predicted_Labels1 = classifier1.predict(images)
            Predicted_Labels2 = classifier2.predict(images)
            Predicted_Labels3 = classifier3.predict(images)
            Predicted_Labels4 = classifier4.predict(images)
            
            P_ensemble=(Predicted_Labels1+Predicted_Labels2+Predicted_Labels3+Predicted_Labels4)

            
            Labels = np.argmax(P_ensemble,1)+1
            for filename, label in zip(filenames, Labels):
                out_file.write('{0},{1}\n'.format(filename, label))
          
          
if __name__ == '__main__':
  tf.app.run()

