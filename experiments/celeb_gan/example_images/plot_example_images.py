import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Set paths
TRAIN_PATH = "experiments/celeb_gan/data/train_dist"
TEST_PATH = "experiments/celeb_gan/data/ipw_taylor_test_data/taylor_random_68"

# Load labels
labels = pd.read_csv(os.path.join(TRAIN_PATH, "labels.csv"))


# Function to load images
def get_img(path, index=0):
    return Image.open(os.path.join(path, "images", f"image_{index:06}.png"))


# For each of the selected attributes, plot first 10 images
for name in ["Bald", "Wearing_Lipstick", "Male", "Smiling"]:
    plt.figure(figsize=(8, 3.6))
    plt.suptitle(name.replace("_", " "), size=20)
    img_idx = labels[labels[name] == 1].index
    for i, idx in enumerate(img_idx[:10]):
        plt.subplot(2, 5, i + 1)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0.05, hspace=0)
        img = get_img(TRAIN_PATH, idx)
        plt.imshow(img)
        plt.axis("off")
    plt.savefig(f"experiments/celeb_gan/example_images/{name}.png")
    plt.clf()

# Plot random sample of training and test images
np.random.seed(1)
for name, path in zip(("Training", "Test"), (TRAIN_PATH, TEST_PATH)):
    plt.figure(figsize=(8, 3.6))
    plt.suptitle(name, size=20)
    for i, idx in enumerate(np.random.choice(5000, 10)):
        plt.subplot(2, 5, i + 1)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0.05, hspace=0)
        img = get_img(path, idx)
        plt.imshow(img)
        plt.axis("off")
    plt.savefig(f"experiments/celeb_gan/example_images/{name}.png")
    plt.clf()
