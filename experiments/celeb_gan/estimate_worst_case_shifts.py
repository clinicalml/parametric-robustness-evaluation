import numpy as np
import time
import pandas as pd
import torchvision.models as models
from tqdm import tqdm
import pickle
import torch
import os
from experiments.celeb_gan.preprocessing import CelebADataset
from torch.utils.data import DataLoader
import experiments.celeb_gan.ipw_functions as ipw
from source.shift_gradients import ShiftLossEstimator
import argparse

def eval_model(model, dataloader, device='cuda', raw_output=False):
    acc       = torch.empty((0, 1)).to(device)
    # Iterate over dataf
    with torch.no_grad():
        model.eval()
        for inputs, labels, metadata in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Compute predictions
            outputs = model(inputs)
            
            labels = labels.view(-1, 1)
            preds = outputs.argmax(dim=1).view(-1, 1)
            acc = torch.cat([acc, 1.0*(preds == labels)], dim=0)

    if raw_output:
        return acc
    else:
        return acc.mean().item()

def get_dataloader(path, batch_size=256):
    dataset = CelebADataset(meta_path=os.path.join(path), img_path=os.path.join(path, "images"), target_name='Male', train_frac=0.0, val_frac=0.0)
    splits = ['test']
    dataloaders = {split: DataLoader(dataset=ds, 
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=16) for split, ds in dataset.get_splits(splits).items()}
    return dataloaders['test']


def grad_s(delta=None, Z=None):
    if Z is not None:
        n, d = Z.shape
        out = torch.zeros((n, 2**d, 1))
        groups = (Z @ torch.diag(torch.tensor([2.0**j for j in range(Z.shape[1])])).sum(axis=1)).long()
        out[torch.arange(n), groups] = 1
    else : 
        n, d = 1, 0
        out = torch.ones((n, 2**d, 1))
    return out

def hess_s(delta=None, Z=None):
    if Z is not None:
        n, d = Z.shape
        out = torch.zeros((n, 2**d, 2**d, 1))
        groups = (Z @ torch.diag(torch.tensor([2.0**j for j in range(Z.shape[1])])).sum(axis=1)).long()
        out[torch.arange(n), groups, groups] = 1
    else:
        n, d = 1, 0
        out = torch.ones((n, 2**d, 2**d, 1))
    return out


def sufficient_statistic(W=None, Z=None):
    return W

def get_taylor(data, cpd):
    Z_list = []
    W_list = []
    var_names = []
    children_sorted = sorted(data.drop('Male', axis=1).columns)

    for child in children_sorted:
        var = data[child]
        parents = cpd[child]['Parents']
        d = len(parents.keys())
        parents_sorted = sorted(parents.keys())
        if d == 0:
            var_names.append(f"{child} | No parents")
        else:
            # If considering shifts to individual parents, this, admittedly absurd code, maps parents names to the deltas. Only for printing though. 
            var_names += [f"{child} | " + ", ".join([f"{pa}={val}" for pa, val in zip (parents_sorted, map(int, reversed(list(f"{j:0{d}b}"))))]) for j in range(2**d)]
        W_list.append(torch.from_numpy(var.values).float().view(-1, 1))
        Z = None if not parents else torch.from_numpy(data[parents_sorted].values).float()
        Z_list.append(Z)
    
    E_taylor = sle.forward(acc, W=W_list, Z=Z_list, shift_strength=args.shift_strength, worst_case_is_larger_loss=False).item()
    delta_taylor = sle.delta.numpy().ravel()

    return E_taylor, delta_taylor

# Set paths
DATA_PATH = 'experiments/celeb_gan/data/ipw_taylor_comparison'
MODEL_PATH = 'experiments/celeb_gan/models/resnet_finetuned.pt'
SAVE_PATH = 'experiments/celeb_gan/compare_ipw_taylor_optim'
TRAIN_PATH = 'experiments/celeb_gan/data/train_dist/'


parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='trust-constr')
parser.add_argument('--shift_strength', type=float, default=2.0)
parser.add_argument('--n_sims', type=int, default=100)


if __name__ == '__main__':
    with open(os.path.join(TRAIN_PATH, 'cpd.pkl'), 'rb') as f:
        CPD_0 = pickle.load(f)

    args = parser.parse_known_args()[0]
    


    # Load model
    model = models.wide_resnet50_2()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(MODEL_PATH))
    device = "cuda"
    model.to(device)
    batch_size = 256

    # Set up shift gradient
    sle = ShiftLossEstimator(sufficient_statistic, grad_s, hess_s)

    reps = np.arange(args.n_sims)

    deltas_ipw = []
    deltas_taylor = []
    other_results = []

    for j in tqdm(reps):
        # Evaluate accuracy of model
        
        acc = eval_model(model, get_dataloader(os.path.join(DATA_PATH, 'random_{}'.format(j))), device=device, raw_output=True)
        acc = acc.cpu().numpy().ravel()
        data = pd.read_csv(os.path.join(DATA_PATH, 'random_{}'.format(j), 'labels.csv')).drop("file_path", axis=1)

        # Compute IPW    
        start_ipw = time.time()
        delta_ipw = ipw.optimize_ipw(data, acc, CPD_0, args)
        E_ipw = ipw.ipw(delta_ipw, data, acc, CPD_0)
        end_ipw = time.time()
        elapsed_ipw = end_ipw - start_ipw


        # Compute Taylor
        start_taylor = time.time()
        E_taylor, delta_taylor = get_taylor(data, CPD_0)
        end_taylor = time.time()
        elapsed_taylor = end_taylor - start_taylor

        # Compute IPW on Taylor Delta
        ipw_on_taylor = ipw.ipw(delta_taylor, data, acc, CPD_0)

        deltas_ipw.append(delta_ipw)
        deltas_taylor.append(delta_taylor)
        other_results.append({"E_ipw": E_ipw, "E_taylor": E_taylor, "elapsed_ipw": elapsed_ipw, "elapsed_taylor": elapsed_taylor, "Training acc": acc.mean(), "IPW on Taylor": ipw_on_taylor})


    df_deltas_ipw = pd.DataFrame(deltas_ipw)
    df_deltas_taylor = pd.DataFrame(deltas_taylor)
    df_other_results = pd.DataFrame(other_results)

    # Check if path exists
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    df_deltas_ipw.to_csv(os.path.join(SAVE_PATH, 'deltas_ipw.csv'))
    df_deltas_taylor.to_csv(os.path.join(SAVE_PATH, 'deltas_taylor.csv'))
    df_other_results.to_csv(os.path.join(SAVE_PATH, 'other_results.csv'))