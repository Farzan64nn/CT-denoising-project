# CT-denoising-project



Run the make_data.py to prepare the AAPM dataset for training and test, and then run each of the following networks independently to get the results.

LSGAN => LSGAN model with SSIM, L1 and aversarial loss functions

CNN_MSE=> CNN model with MSE loss function

CNN_VGG=> CNN model with Perceptual loss function

CNN_MVGG=> CNN model with joint loss function

RED-CNN=> AE model with only MSE loss function

GAN=> GAN framework and only adverserial loss function

GAN=> GAN framework with only MSE loss function

GAN-VGG=> GAN framework with joint (MSE and Perceptual) loss function

WGAN=> WGAN framework (considering Wasserstein distance) with MSE loss function

WGAN-VGG=> WGAN framework (considering Wasserstein distance) with joint loss function

To run the above codes, you need to set the path for the train and test datasets. 

utils.py includes all the functions to manipulate the CT image such as reading, excluding air.

The codes are written in Tensorflow 2.5 with Keras backend.
