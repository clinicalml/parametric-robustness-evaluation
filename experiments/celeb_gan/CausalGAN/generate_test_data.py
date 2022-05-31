from __future__ import print_function
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from IPython.core import debugger
import os
debug = debugger.Pdb().set_trace
from main import get_trainer, main
from generate_training_data import CPD_0, shift, generate_data
import argparse

PATH = '../data/celeb_gan/test_dist/'

def make_shift(cpd, children, deltas, parents=None):
    if parents is None:
        parents = ["Base" for _ in range(len(children))]
    cpd = deepcopy(cpd)
    for child, parent, delta in zip(children, parents, deltas):
        cpd = shift(deepcopy(cpd), child, parent, delta)
    return cpd

parser = argparse.ArgumentParser()
parser.add_argument('--folder_name', type=str, default='default')
parser.add_argument('--M', type=int, default=30)
parser.add_argument('--N', type=int, default=500)


if __name__ == "__main__":
    
    # Load CLI arguments
    args = parser.parse_known_args()[0]
    results_path = os.path.join("../experiments/celeb_gan/results/", args.folder_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    M = args.M
    N = args.N
    
    # Initialize model
    trainer = get_trainer()
    sess = trainer.sess
    cc=trainer.cc
    if hasattr(trainer,'model'):
        model=trainer.model
    main(trainer)

    # Make simultaneous shift table
    df = pd.read_csv(os.path.join(results_path, 'simultaneous_deltas.csv'))
    cpd = make_shift(CPD_0, df['A_names'], df['delta'])
    generate_data(sess, trainer, cc, model, N, M, cpd, os.path.join(PATH, 'simultaneous'))

    # Generate marginal shift data
    df = pd.read_csv(os.path.join(results_path, "marginal_sensitivities.csv"))
    for index, row in tqdm(df.iterrows(), total=len(df)):
        child = row['Feature']
        delta = row['delta']
        child_delta = str(child) + str(delta)

        cpd = shift(deepcopy(CPD_0), child, "Base", delta)
        generate_data(sess, trainer, cc, model, N, M, cpd, os.path.join(PATH, "{}".format(child_delta)))
    
    sess.close()