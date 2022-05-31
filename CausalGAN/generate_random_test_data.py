from __future__ import print_function
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from IPython.core import debugger
import os
debug = debugger.Pdb().set_trace
from main import get_trainer, main
from generate_training_data import CPD_0, shift, generate_data

PATH = '../data/celeb_gan/random_test_dist/'

def make_shift(cpd, children, deltas, parents=None):
    if parents is None:
        parents = ["Base" for _ in range(len(children))]
    cpd = deepcopy(cpd)
    for child, parent, delta in zip(children, parents, deltas):
        cpd = shift(deepcopy(cpd), child, parent, delta)
    return cpd


if __name__ == "__main__":
    N = 500
    M = 1
    n_random = 400
    
    # Initialize model
    trainer = get_trainer()
    sess = trainer.sess
    cc=trainer.cc
    if hasattr(trainer,'model'):
        model=trainer.model
    main(trainer)
    
    # Simulate random shift delta
    for j in tqdm(range(n_random)):
        # Simulate random delta and normalize to unit ball
        delta = np.random.normal(size=(len(CPD_0)))
        delta = 2*delta / np.linalg.norm(delta)
        # Make shift table and generate data
        cpd = make_shift(CPD_0, CPD_0.keys(), delta)
        generate_data(sess, trainer, cc, model, N, M, cpd, os.path.join(PATH, 'random_{}'.format(j)))
    
    sess.close()