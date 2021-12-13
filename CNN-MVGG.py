
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

IMG_HEIGHT = None
IMG_WIDTH = None
OUTPUT_CHANNELS = 1
BUFFER_SIZE = 482075
IMAGE_SHAPE = (IMG_HEIGHT , IMG_WIDTH , OUTPUT_CHANNELS)
BATCH_SIZE = 4  
#---------------------Load data and preparation-------------------------------  

def load_data():
    
    with h5py.File('Train40.h5', 'r') as hf:
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
    inputs = layers.Input(shape = IMAGE_SHAPE)          
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = layers.Conv2D(64, (3, 3), padding='same',dilation_rate=(2,2))(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    
    conv3 = layers.Conv2D(64, (3, 3), padding='same',dilation_rate=(3,3))(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    
    conv4 = layers.Conv2D(64, (3, 3), padding='same',dilation_rate=(4,4))(conv3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    
    conv5 = layers.Conv2D(64, (3, 3), padding='same',dilation_rate=(3,3))(conv4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)
    #Residual blocks via concatenate 
    conv6 = layers.Concatenate(axis=3)([conv5,conv2])
    conv6 = layers.Conv2D(64, (3, 3), padding='same',dilation_rate=(2,2))(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)
    #Residual blocks via concatenate
    conv7 = layers.Concatenate(axis=3)([conv6,conv1])
    conv8 = layers.Conv2D(1, (3, 3), padding='same')(conv7)

    penultimate = layers.Add()([conv8,inputs])
    output = layers.Conv2D(1, (3, 3), padding='same')(penultimate)
    # outputs = layers.Concatenate(axis=-1)([output,output,output])
    model = keras.models.Model(inputs=[inputs], outputs=[output] , name="model")
        
    return model

#----------------------------------------------------------------------------
model = define_model()

loss_object = tf.keras.losses.MeanSquaredError()
model_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.9)

#-----------------------------Check point saver-------------------------------


checkpoint_dir = './training_checkpoints3'
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

def measurments(model, test_input, tar):
    prediction = model(test_input, training=True)
    ssim = tf.image.ssim(prediction,tar[...,tf.newaxis], max_val=1.0)
    psnr = tf.image.psnr(prediction,tar[...,tf.newaxis], max_val=1.0)
    ssim = tf.reduce_mean(ssim)
    psnr = tf.reduce_mean(psnr)
    return ssim , psnr  
#--------------------------------Load VGG model------------------------------- 

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape = [40,40,3])
vgg.trainable = False
content_layers = 'block5_conv4'

lossModel = tf.keras.models.Model([vgg.input], vgg.get_layer(content_layers).output, name = 'vggL')

#--------------------------------Loss Function -------------------------------

def lossVGG(X,Y):
    Xt = tf.keras.applications.vgg19.preprocess_input(X*255)
    Yt = tf.keras.applications.vgg19.preprocess_input(Y*255)
    vggX = lossModel(Xt)
    vggY = lossModel(Yt)
    return tf.reduce_mean(tf.square(vggY-vggX))



#------------------------------Training----------------------------------------

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as model_tape:
    model_output = model(input_image, training=True)
    model_output_vgg = tf.image.grayscale_to_rgb(model_output) 
    target_vgg = tf.image.grayscale_to_rgb(target[...,tf.newaxis])      
    feature_loss = lossVGG(model_output_vgg, target_vgg)
    feature_loss = (1/(2.0*2.0*512.0)) * feature_loss

    
    l2_loss = loss_object(model_output, target[...,tf.newaxis])

    total_loss = feature_loss*.1 + l2_loss
    model_gradients = model_tape.gradient(total_loss,
                                            model.trainable_variables)
    
    model_optimizer.apply_gradients(zip(model_gradients,
                                            model.trainable_variables))
    
    SSIM , PSNR = measurments(model, input_image, target)

    with summary_writer.as_default():
      tf.summary.scalar('feature_loss', feature_loss, step=step//1000)
      tf.summary.scalar('l2_loss', l2_loss, step=step//1000)
      tf.summary.scalar('total_loss', total_loss, step=step//1000)
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




# Restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Run the trained model on a few examples from the test set
for inp, tar in test_ds.take(5):
  generate_images(model, inp, tar)
  
  
  
  
