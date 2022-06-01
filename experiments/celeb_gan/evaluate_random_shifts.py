import pandas as pd
import torchvision.models as models
from tqdm import tqdm
import torch
import os
from experiments.celeb_gan.estimate_worst_case_shifts import eval_model, get_dataloader
import argparse

# Set paths
DATA_PATH = 'experiments/celeb_gan/data/random_test_dist'
DATA_PATH_31 = 'experiments/celeb_gan/data/random_test_dist_31'
MODEL_PATH = 'experiments/celeb_gan/models/resnet_finetuned.pt'
RESULTS_PATH = 'experiments/celeb_gan/results/'
SAVE_PATH = 'experiments/celeb_gan/plots/'
LOAD_OURS_PATH = 'experiments/celeb_gan/compare_ipw_taylor_optim'

parser = argparse.ArgumentParser()
parser.add_argument('--use_31', type=bool, default=False)
parser.add_argument('--n_random', type=int, default=400)


if __name__ == '__main__':
    args = parser.parse_args()
    data_path = DATA_PATH_31 if args.use_31 else DATA_PATH
    n_random = args.n_random

    # Load model
    model = models.wide_resnet50_2()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(MODEL_PATH))
    device = "cuda"
    model.to(device)
    batch_size = 256

    # Simultaneous
    accs = [eval_model(model, get_dataloader(os.path.join(data_path, 'random_{}'.format(j)))) for j in tqdm(range(n_random))]
    df = pd.DataFrame(accs, columns=['Loss'])
    df.to_csv(os.path.join(SAVE_PATH, f'random_acc{str(31) if args.use_31 else ""}.csv'), index=False)
    df.to_csv(os.path.join(RESULTS_PATH, 'figure5_left.csv'), index=False)
    print(df.round(3))

    # Print the worst-case random delta
    worst_case_df = pd.read_csv(os.path.join(LOAD_OURS_PATH, "results_with_ground_truth_first.csv"))
    worst_case_df.index = worst_case_df['Unnamed: 0']
    number_worse = (df['Loss'] > worst_case_df.loc['E_taylor actual'][1]).mean()
    
    print(f"The worst-case random delta is worse than {100*number_worse:.2f}% of the simulated data")

    