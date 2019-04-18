# Motionganvae

This is the code to the master thesis 'Motion Prediction with the Improved Wasserstein GAN' written at TU Graz. It is based on the code from the paper 'Improved training of Wasserstein GANs': https://github.com/igul222/improved_wgan_training

## There is one file for MNIST experiments:

GAN_mnist_allinone

The different models can be chosen in the code: GAN, cGAN, VAE, EncGAN

## There are multiple files for experiments on moving MNIST dataset:

GAN_movingmnist_allinone  -- without time convolution, with 64x64 image resolution for GAN and cGAN

GAN_movingmnist_allinone_gan32  -- without time convolution, only 32x32 image resolution possible

GAN_movingmnist_allinone_vaefuture  -- for the VAE+ experiment to predict a frame with a VAE

GAN_movingmnist_allinone_time  -- for the time convolution V1 experiments with different number of past frames

GAN_movingmnist_allinone_convtime_v2  -- for the time convolution V2 experiments with 3 past frames

GAN_movingmnist_allinone_convtime_v3  -- for the time convolution V3 experiments with 3 past frames


The functions for loading the MNIST and moving MNIST dataset as well as functions for plotting and saving images can be found in the tflib folder.
