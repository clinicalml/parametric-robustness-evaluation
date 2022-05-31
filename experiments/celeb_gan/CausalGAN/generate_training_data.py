from __future__ import print_function
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import glob
from tqdm import tqdm
from IPython.core import debugger
import pandas as pd
debug = debugger.Pdb().set_trace
import pickle
from main import get_trainer, main
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PATH = '../data/celeb_gan/train_dist/'

# Base conditional probabilities
CPD_0 =  {
    'Young':                {'Base': 0.0, "Parents": dict()},
    'Male':                 {'Base': 0.0, "Parents": dict()},
    'Eyeglasses':           {'Base': 0.0, "Parents": {'Young': -0.4}},
    'Bald':                 {'Base': -3.0, "Parents": {'Male': 3.5, 'Young': -1.0}},
    'Mustache':             {'Base': -2.5, "Parents": {'Male': 2.5, 'Young': -1.0}},
    'Smiling':              {'Base': 0.25, "Parents": {'Male': -0.5, 'Young': 0.5}},
    'Wearing_Lipstick':     {'Base': 3.0, "Parents": {'Male': -5.0, 'Young': -0.5}},
    'Mouth_Slightly_Open':  {'Base': -1.0, "Parents": {'Young': 0.5, 'Smiling': 1.0}},
    'Narrow_Eyes':          {'Base': -0.5, "Parents": {'Male': 0.3, 'Young': 0.2, 'Smiling': 1.0}}
}

# Simulation
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sim_labels(N, CPD):
    labels = {}
    list_of_variables = CPD.items()
    for child, probs in list_of_variables:
        defer = False
        # Check if all parents are already simulated
        for parent in probs['Parents'].keys():
            if parent not in labels:
                # If not, defer this child
                list_of_variables.append((child, probs))
                defer = True
                break
        if not defer:
            logprobs = probs['Base']*np.ones((N,1))
            for parent, coef in probs['Parents'].items():
                logprobs += coef*labels[parent]
            labels[child] =  1.0*(np.random.uniform(size=(N,1)) < sigmoid(logprobs))
    return labels


def pack_labels(labels):
    return np.concatenate([labels[child] for child in labels.keys()], axis=1)

def shift(CPD, child, parent="Base", shift_size=0.0):
    CPD[child][parent] += shift_size
    return CPD

def generate_data(sess, trainer, cc, model, N, M, cpd, path):
    # Ensure that path exists
    if not os.path.exists(os.path.join(path, 'images')):
        os.makedirs(os.path.join(path, 'images'))

    # First delete old images
    files = glob.glob(path + '*.png')
    for f in files:
        os.remove(f)

    with open(os.path.join(path, 'cpd.pkl'), 'wb') as f:
        pickle.dump(cpd, f)
    
    labels = None
    for j in range(M):
        labels_ = sim_labels(N, cpd)
        feed_dict={cc.label_dict[k]:v for k,v in labels_.iteritems()}
        feed_dict[trainer.batch_size]=N
        images=sess.run(model.G,feed_dict)

        for i, image in enumerate(images):
            Image.fromarray(image.astype(np.uint8)).save(os.path.join(path, "images/image_{}.png".format("%06d" % (j*N + i))))

        tf.reset_default_graph()
        # Store labels in pandas dataframe
        if labels is None:
            labels = pd.DataFrame({k: v.ravel().tolist() for k,v in labels_.items()})
        else:
            labels = pd.concat([labels, pd.DataFrame({k: v.ravel().tolist() for k,v in labels_.items()})], ignore_index=True)
    labels['file_path'] = ["image_{}.png".format("%06d" % i) for i in range(len(labels))]
    labels.to_csv(os.path.join(path, 'labels.csv'), index=False)


N = 500
M = 40
if __name__ == "__main__":
    # First delete old images
    files = glob.glob(PATH + '*.png')
    for f in files:
        os.remove(f)

    # Initialize model
    trainer = get_trainer()
    sess = trainer.sess
    cc=trainer.cc
    if hasattr(trainer,'model'):
        model=trainer.model
    main(trainer)

    # Generate data
    generate_data(sess, trainer, cc, model, N, M, CPD_0, PATH)

    sess.close()

