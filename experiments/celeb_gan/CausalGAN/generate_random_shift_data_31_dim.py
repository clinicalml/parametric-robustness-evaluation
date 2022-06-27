from __future__ import print_function
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from IPython.core import debugger
import os
import pandas as pd
debug = debugger.Pdb().set_trace
from main import get_trainer, main
import argparse
import tensorflow as tf
from generate_training_data import CPD_0, generate_data
from PIL import Image
import glob
import pickle

DATA_PATH = '../data/train_dist'
PATH = '../data/random_test_dist_31/'

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sim_labels(n, data, cpd, delta=None):
    # Split the same way as IPW does
    children_sorted = sorted(data.drop("Male", axis=1).columns)
    splits = np.cumsum([2**len(cpd[child]['Parents']) for child in children_sorted])
    deltas = np.split(delta, splits)[:-1]
    child_deltas = {child: delta for child, delta in zip(children_sorted, deltas)}
    child_deltas['Male'] = np.array([0])
    labels = {}
    list_of_variables = list(cpd.items())

    for child, probs in list_of_variables:
        defer = False
        delta_child = child_deltas[child]
        parents = probs['Parents']
        parents_sorted = sorted(parents.keys())
        # Check if all parents are already simulated
        for parent in parents_sorted:
            if parent not in labels:
                # If not, defer this child
                list_of_variables.append((child, probs))
                defer = True
                break
        if not defer:
            if parents:
                d = len(parents)
                groups = (np.matmul(np.array([labels[parent] for parent in parents_sorted]).T, np.array([2**j for j in range(d)])).astype(int))
            else:
                groups = np.zeros(n).astype(int)
            eta_obs = probs['Base'] + delta_child[groups] + np.sum([labels[parent]*parents[parent] for parent in parents_sorted], axis=0)                        
            labels[child] =  1.0*(np.random.uniform(size=(n)) < sigmoid(eta_obs))
    
    labels = {k: v.reshape(-1, 1) for k,v in labels.items()}

    return labels

def generate_data(sess, trainer, cc, model, N, M, cpd, path, delta=None, data=None):
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
        labels_ = sim_labels(N, data, cpd, delta)
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

parser = argparse.ArgumentParser()
parser.add_argument('--n_random', type=int, default=400)
parser.add_argument('--N', type=int, default=500)
parser.add_argument('--M', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_known_args()[0]
    N = args.N
    M = args.M
    seed = args.seed
    n_random = args.n_random
    
    # Initialize model
    trainer = get_trainer()
    sess = trainer.sess
    cc=trainer.cc
    if hasattr(trainer,'model'):
        model=trainer.model
    main(trainer)
    
    # Load saved data (dummy, only to ensure right ordering of delta)
    data = pd.read_csv(os.path.join(DATA_PATH, 'labels.csv')).drop("file_path", axis=1)

    np.random.seed(seed)

    for j in tqdm(range(n_random)):
        cpd = deepcopy(CPD_0)    
        delta = np.random.normal(size=31)
        delta = 2*delta / np.linalg.norm(delta)

        # Make shift table and generate data
        generate_data(sess, trainer, cc, model, N, M, cpd, os.path.join(PATH, 'random_{}'.format(j)), delta, data)

    sess.close()
