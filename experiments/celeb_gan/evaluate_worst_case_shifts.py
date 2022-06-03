import experiments.celeb_gan.ipw_functions as ipw
import pandas as pd
import torchvision.models as models
from tqdm import tqdm
import torch
import os
import pickle
from experiments.celeb_gan.estimate_worst_case_shifts import eval_model, get_dataloader

# Set paths
PATH_TEST   = 'experiments/celeb_gan/data/ipw_taylor_test_data/'
SAVE_PATH   = 'experiments/celeb_gan/compare_ipw_taylor_optim'
MODEL_PATH  = 'experiments/celeb_gan/models/resnet_finetuned.pt'
TRAIN_PATH = 'experiments/celeb_gan/data/train_dist/'
RESULTS_PATH = 'experiments/celeb_gan/results/'
if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)

with open(os.path.join(TRAIN_PATH, 'cpd.pkl'), 'rb') as f:
    CPD_0 = pickle.load(f)

# Load model
model       = models.wide_resnet50_2()
num_ftrs    = model.fc.in_features
model.fc    = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(MODEL_PATH))
device      = "cuda"
model.to(device)
batch_size  = 256

df_deltas_ipw       = pd.read_csv(os.path.join(SAVE_PATH, 'deltas_ipw.csv'))
df_deltas_taylor    = pd.read_csv(os.path.join(SAVE_PATH, 'deltas_taylor.csv'))
df                  = pd.read_csv(os.path.join(SAVE_PATH, 'other_results.csv'))

# Simultaneous
n_random = df.shape[0]
accs_ipw = [eval_model(model, get_dataloader(os.path.join(PATH_TEST, 'ipw_random_{}'.format(j)))) for j in tqdm(range(n_random))]
accs_taylor = [eval_model(model, get_dataloader(os.path.join(PATH_TEST, 'taylor_random_{}'.format(j)))) for j in tqdm(range(n_random))]

df['E_ipw actual'] = accs_ipw
df['E_taylor actual'] = accs_taylor
df.to_csv(os.path.join(SAVE_PATH, 'results_with_ground_truth.csv'), index=False)
df.to_csv(os.path.join(RESULTS_PATH, 'figure5_right.csv'), index=False)

# Save first row for table
df.loc[0,:].to_csv(os.path.join(SAVE_PATH, 'results_with_ground_truth_first.csv'))
df.loc[0,:].to_csv(os.path.join(RESULTS_PATH, 'table1_right.csv'))

# Print the worst-case random delta
print(f"IPW error: {(df['E_ipw actual'] - df['E_ipw']).abs().mean():.3f}")
print(f"Taylor error: {(df['E_taylor actual'] - df['E_taylor']).abs().mean():.3f}")

  