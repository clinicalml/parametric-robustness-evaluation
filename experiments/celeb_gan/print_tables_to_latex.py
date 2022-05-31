import pandas as pd
import torch
import os
import pickle
import numpy as np
device = "cpu"

def sigmoid(x):
    return 1/(1 + np.exp(-x))

FOLDER_NAME = 'default'

# Get tables of probabilities
df_train = pd.read_csv(os.path.join("experiments/celeb_gan/variables", FOLDER_NAME, "metas.csv"))
df_train = df_train.mean(axis=0)
df_test = pd.read_csv("data/celeb_gan/test_dist/simultaneous/labels.csv")
df_test = df_test.drop("file_path", axis=1).mean(axis=0)
df_shift = pd.read_csv(os.path.join("experiments/celeb_gan/results", FOLDER_NAME,"simultaneous_deltas.csv"))[['A_names', 'delta']]
df_shift.index = df_shift['A_names']
df_stats = pd.concat([df_train, df_test, df_shift.drop("A_names", axis=1)], axis=1)
df_stats.columns=["$\P(W_i = 1)$", "$\P_\delta(W_i = 1)$", "Shift $\delta_i$"]
df_stats['Characteristic $W_i$'] = df_stats.index
cols = list(df_stats)
cols.insert(0, cols.pop(cols.index('Characteristic $W_i$')))
df_stats = df_stats.reindex(columns=cols)
df_stats['Characteristic $W_i$'] = [s.replace("_", " ") for s in df_stats['Characteristic $W_i$']]
df_stats.drop("Male").round(2).to_latex(f'../shift_gradients_overleaf_clone/tables/{FOLDER_NAME}_celeb_gan_table.tex', index=False, escape=False)

# Print IPW-vs-Taylor optimization speeds
df = pd.read_csv("experiments/celeb_gan/compare_ipw_taylor_optim/results_with_ground_truth.csv")
print(f"{df['elapsed_ipw'].mean():.3f} s")
print(f"{df['elapsed_taylor'].mean():.3f} s")


# Get delta individual
# Check if the file exists
if os.path.isfile(os.path.join("experiments/celeb_gan/results", FOLDER_NAME,"simultaneous_deltas_individual.csv")):
    df = pd.read_csv(os.path.join("experiments/celeb_gan/results", FOLDER_NAME,"simultaneous_deltas_individual.csv"))

    TRAIN_PATH = 'data/celeb_gan/train_dist/'
    with open(os.path.join(TRAIN_PATH, 'CPD_0.pkl'), 'rb') as f:
        CPD_0 = pickle.load(f)

    df = df[['A_names', 'delta']]
    df['A_names'] = [name.replace("_", " ") for name in df['A_names']]
    df.columns = ['Conditional', '$\delta_i$']
    df['Abs'] = df['$\delta_i$'].abs()
    # Sort by delta
    df = df.sort_values(by='Abs', ascending=False).drop("Abs", axis=1)
    df['Conditional'] = [s.replace("=1", "$=1$").replace("=0", "$=0$") for s in df['Conditional']]
    df.round(3).to_latex(f'../shift_gradients_overleaf_clone/tables/{FOLDER_NAME}_celeb_gan_table_individual.tex', index=False, escape=False)
    
    top_5 = df[:5]


    # Handcraft
    cpd = CPD_0['Wearing_Lipstick']
    eta1 = cpd['Base'] + 1*cpd['Parents']['Male'] + 1*cpd['Parents']['Young']
    eta2 = cpd['Base'] + 0*cpd['Parents']['Male'] + 1*cpd['Parents']['Young']
    eta4 = cpd['Base'] + 1*cpd['Parents']['Male'] + 0*cpd['Parents']['Young']
    cpd = CPD_0['Bald']
    eta3 = cpd['Base'] + 0*cpd['Parents']['Male'] + 0*cpd['Parents']['Young']
    eta5 = cpd['Base'] + 1*cpd['Parents']['Male'] + 1*cpd['Parents']['Young']
    top_5['$\P_{\delta}(W|\PA(W))$'] = [np.round(sigmoid(eta + delta),3) for eta, delta in zip([eval(f"eta{j}") for j in range(1,6)], df['$\delta_i$'])]

    for a, b in [("Male$=1$", "Male"), ("Male$=0$", "Female"), ("Young$=1$", "Young"), ("Young$=0$", "Old")]:
        top_5['Conditional'] = [x.replace(a, b) for x in top_5['Conditional']]
    top_5.round(3).to_latex(f'../shift_gradients_overleaf_clone/tables/{FOLDER_NAME}_celeb_gan_table_individual_top5.tex', index=False, escape=False)

# Results table
Y = torch.load(os.path.join('experiments/celeb_gan/variables', FOLDER_NAME, 'targets.pt'), map_location=device).view(-1, 1)
preds = torch.load(os.path.join('experiments/celeb_gan/variables', FOLDER_NAME, 'preds.pt'), map_location=device)
acc = (preds == Y).float().mean().item()

df = pd.read_csv(os.path.join('experiments/celeb_gan/results', FOLDER_NAME, "simultaneous_acc.csv"))
already_enhanced = "(Observed accuracy)" in df.columns
worst_case_at_delta = df['Ground truth'][0]
if not already_enhanced:
    df.index = ['Estimate']
    # cols = list(df)
    # cols.insert(0, cols.pop(cols.index('tmp')))
    # df = df.loc[:, cols]
    df.columns = ['Taylor estimate ($\\taylor$)', 'IPW estimate ($\ipw$)', 'Ground truth test acc. ($\E_\delta[\ell_\gamma]$)']
    df['(Training acc ($\E[\ell_\gamma]$))'] = acc
    # df.to_csv(os.path.join('experiments/celeb_gan/results', FOLDER_NAME, "simultaneous_acc.csv"), index=False)
df.round(3).to_latex('../shift_gradients_overleaf_clone/tables/celeb_gan_table_sensitivities.tex', index=False, escape=False)

# Results table for multiple vars
Y = torch.load(os.path.join('experiments/celeb_gan/variables', "test-multiple-var", 'targets.pt'), map_location=device).view(-1, 1)
preds = torch.load(os.path.join('experiments/celeb_gan/variables', "test-multiple-var", 'preds.pt'), map_location=device)
acc = (preds == Y).float().mean().item()

# Results for 31 dimensional
df = pd.read_csv(os.path.join('experiments/celeb_gan/compare_ipw_taylor_optim/results_with_ground_truth_first.csv'))
df.index = ['Estimate']
df.drop("Unnamed: 0", axis=1, inplace=True)
df = df[["E_taylor", "IPW on Taylor", "E_taylor actual", "Training acc",]]
df.columns = ['Taylor estimate ($\\taylor$)', 'IPW estimate ($\ipw$)', 'Ground truth test acc. ($\E_\delta[\ell_\gamma]$)', "Training acc. ($\E[\ell_\gamma]$)"]
df.round(3).to_latex('../shift_gradients_overleaf_clone/tables/celeb_gan_table_sensitivities.tex', index=False, escape=False)

# Compare estimates IPW and Taylor in 31 dim
df = pd.read_csv(os.path.join('experiments/celeb_gan/compare_ipw_taylor_optim/results_with_ground_truth.csv'))
df['Estimation error (IPW)'] = (df['E_ipw'] - df['E_ipw actual']).abs()
df['Estimation error (Taylor)'] = (df['E_taylor'] - df['E_taylor actual']).abs()

df.mean(axis=0)


# Results for marginal shifts
df = pd.read_csv('experiments/celeb_gan/output_tables/marginal_sensitivities.csv')
df.round(3).to_latex("../shift_gradients_overleaf_clone/tables/celeb_gan_table_marginal_sensitivities.tex", index=False, escape=False)

