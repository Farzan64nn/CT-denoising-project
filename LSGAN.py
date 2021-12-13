
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
BUFFER_SIZE = 482075         # The total number of training patches
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

    return trainX, trainY, testX, testY


# Normalizing the images to [-1, 1]
def normalize(x, y , n , m):
  x = (x *2) - 1
  y = (y *2) - 1
  n = (n *2) - 1
  m = (m *2) - 1

  return (x, y) , (n, m)
x_train, y_train, x_test, y_test = load_data()
(x_train, y_train), (x_test, y_test) = normalize(x_train, y_train, x_test, y_test)


train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)


#----------------------------Make the generator-------------------------------

def generator_model():
    def ConvLayers_middle_block(x):
        x = layers.Conv2D(96, kernel_size=3, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(96, kernel_size=5, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        return x
        
        
    inputs = layers.Input(shape = IMAGE_SHAPE)          
    x = layers.Conv2D(32, (7, 7), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(96, (5, 5), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    for i in range(5):
        y = ConvLayers_middle_block(x)
        x = layers.Subtract()([x,y])
        
    x = layers.Conv2D(96, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(1, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    output = layers.Subtract()([inputs,x])
        
    model = keras.models.Model(inputs=[inputs], outputs=[output] , name="generator")
        
    return model
#----------------------------Make the discriminator----------------------------

def discriminator_model():
  
    inputs = layers.Input(shape = [40,40,1])
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding="same")(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(256)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(1)(x)
    output = layers.Activation('relu')(x)
    model = keras.models.Model(inputs=[inputs], outputs=[output] , name="discriminator") 
    return model


#-----------------------------------------------------------------------------

generator = generator_model()
discriminator = discriminator_model()

#--------------------Define the generator loss--------------------------------
alpha = 0.005
beta = 0.995
gama = 0.95
def generator_loss(disc_generated_output, gen_output, tar):
    
    # print(tf.shape(target))
    gan_loss = 0.5 * tf.reduce_mean(tf.square(disc_generated_output - tf.ones_like(disc_generated_output)))
    l1_loss = tf.reduce_mean(tf.abs(gen_output - tar[...,tf.newaxis]))
    ssim = tf.image.ssim(gen_output, tar[...,tf.newaxis], max_val=1.0) 
    ssim_loss = 1.0 - tf.reduce_mean(ssim)
    total_gen_loss = alpha * gan_loss + beta * l1_loss + gama * ssim_loss
    return total_gen_loss, gan_loss, l1_loss, ssim_loss


#--------------------Define the discriminator loss-----------------------------
def discriminator_loss(real_img, fake_img):
    d1 = 0.5 * tf.reduce_mean(tf.square(real_img - tf.ones_like(real_img)))
    d2 = 0.5 * tf.reduce_mean(tf.square(fake_img))
    return d1 + d2

# learning_rate = 1e-5 to 1e-6 ?
generator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5, beta_2=0.9)



#-----------------------------Check point saver-------------------------------


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generaor_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

#-----------------------------Display the result------------------------------
def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
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
#------------------------------Training----------------------------------------
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(images, target, step):
    
    
    with tf.GradientTape() as disc_tape:
      generated_images = generator(images, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)
      
      
      disc_loss = discriminator_loss(real_output, fake_output)
      

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    with tf.GradientTape() as gen_tape:
        generated_images = generator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_total_loss, gen_gan_loss, gen_l1_loss, gen_ssim_loss = generator_loss(fake_output, generated_images, target)

    gradients_of_generator = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    SSIM , PSNR = measurments(generator, images, target)

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
        tf.summary.scalar('gen_ssim_loss', gen_ssim_loss, step=step//1000)
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

      generate_images(generator, example_input, example_target)
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
  generate_images(generator, inp, tar)
  
  