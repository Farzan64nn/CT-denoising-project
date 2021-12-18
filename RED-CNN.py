# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:57:33 2021

@author: fnikneja
"""

import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import time
from IPython import display

IMG_WIDTH = None
IMG_WIDTH = None
OUTPUT_CHANNELS = 1
BUFFER_SIZE = 482075
IMAGE_SHAPE = (IMG_WIDTH , IMG_WIDTH , OUTPUT_CHANNELS)
BATCH_SIZE = 4    
#---------------------Load data and preparation-------------------------------  

def load_data():
    
    with h5py.File('Train55.h5', 'r') as hf:
            trainX = np.array(hf.get('data'))
            trainY = np.array(hf.get('label'))       

    with h5py.File('Test512.h5', 'r') as hf:
            testX = np.array(hf.get('data'))
            testY = np.array(hf.get('label'))                                                      

    return (trainX, trainY), (testX, testY)

(x_train, y_train), (x_test, y_test) = load_data()


train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)


#----------------------------Build the model----------------------------------

def define_model():
    
    inputs = layers.Input(shape = [None,None,1])
    x1 = layers.Conv2D(96, kernel_size=5, strides=1, padding="valid")(inputs)
    x1 = layers.Activation('relu')(x1)
    x2 = layers.Conv2D(96, kernel_size=5, strides=1, padding="valid")(x1)
    x2 = xx8 = layers.Activation('relu')(x2)
    x3 = layers.Conv2D(96, kernel_size=5, strides=1, padding="valid")(x2)
    x3 = layers.Activation('relu')(x3)
    x4 = layers.Conv2D(96, kernel_size=5, strides=1, padding="valid")(x3)
    x4 = xx6 = layers.Activation('relu')(x4)
    x5 = layers.Conv2D(96, kernel_size=5, strides=1, padding="valid")(x4)
    x5 = layers.Activation('relu')(x5)
    x6 = layers.Conv2DTranspose(96, kernel_size=5, strides=1, padding="valid")(x5)
    x6 += xx6
    x6 = layers.Activation('relu')(x6)
    x7 = layers.Conv2DTranspose(96, kernel_size=5, strides=1, padding="valid")(x6)
    x7 = layers.Activation('relu')(x7)
    x8 = layers.Conv2DTranspose(96, kernel_size=5, strides=1, padding="valid")(x7)
    x8 += xx8
    x8 = layers.Activation('relu')(x8)
    x9 = layers.Conv2DTranspose(96, kernel_size=5, strides=1, padding="valid")(x8)
    x9 = layers.Activation('relu')(x9)
    x10 = layers.Conv2DTranspose(1, kernel_size=5, strides=1, padding="valid")(x9)
    x10 += inputs
    output = layers.Activation('relu')(x10)
    
    model = keras.models.Model(inputs=[inputs], outputs=[output] , name="RED-CNN") 
            
    return model

#----------------------------------------------------------------------------
model = define_model()

loss_object = tf.keras.losses.MeanSquaredError()
model_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.9)

#-----------------------------Check point saver-------------------------------


checkpoint_dir = './training_checkpoints1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model_optimizer=model_optimizer,
                                 model=model)

#-----------------------------Display the result------------------------------
def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    plt.imshow(display_list[i], cmap='gray')
    plt.axis('off')
  plt.show()

#-------------------------Calculate SSIM and PSNR------------------------------

def measurments(model_out, tar):
    ssim = tf.image.ssim(model_out,tar[...,tf.newaxis], max_val=1.0)
    psnr = tf.image.psnr(model_out,tar[...,tf.newaxis], max_val=1.0)
    ssim = tf.reduce_mean(ssim)
    psnr = tf.reduce_mean(psnr)
    return ssim , psnr

#------------------------------Training----------------------------------------

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as model_tape:
    model_output = model(input_image, training=True)
        
    l2_loss = loss_object(model_output, target)
    model_gradients = model_tape.gradient(l2_loss,
                                            model.trainable_variables)
    
    model_optimizer.apply_gradients(zip(model_gradients,
                                            model.trainable_variables))
    
    SSIM , PSNR = measurments(model_output, target)
    
    
    with summary_writer.as_default():
      tf.summary.scalar('l2_loss', l2_loss, step=step//1000)
      tf.summary.scalar('SSIM', SSIM, step=step//1000)
      tf.summary.scalar('PSNR', PSNR, step=step//1000)

      



def fit(train_dataset, test_dataset, steps):
  example_input, example_target = next(iter(test_dataset.take(1)))

  for step, (input_image, target) in train_dataset.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      start = time.time()
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start} sec\n')

      generate_images(model, example_input, example_target)
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)
# Save (checkpoint) the model every 5k steps
    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)      
      
      
fit(train_ds, test_ds, steps=100000)      


#----------------------------------Test----------------------------------------

# Restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Run the trained model on a few examples from the test set
for inp, tar in test_ds.take(5):
  generate_images(model, inp, tar)
  



