#!/bin/sh

# Get CelebA data (due to quirks in the CausalGAN code, it will complain if the data isn't around, even for our pre-trained version)
cd experiments/celeb_gan
mkdir -p CausalGAN/data/data/celebA
cd CausalGAN/data/data/celebA
kaggle datasets download jessicali9530/celeba-dataset --unzip
cd ../../../../../..

# Unzip the checkpoints of our pre-trained CausalGAN
if [ -d experiments/celeb_gan/CausalGAN/logs ]; then
  rm experiments/celeb_gan/CausalGAN/logs -rf
fi

unzip -qq experiments/celeb_gan/CausalGAN/gan.zip -d experiments/celeb_gan/CausalGAN/logs 
