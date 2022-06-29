import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


DIR = "experiments/celeb_gan"
LATEX_DIR = "experiments/celeb_gan/latex"

if not os.path.exists(f"{LATEX_DIR}/figures"):
    os.makedirs(f"{LATEX_DIR}/figures")

if not os.path.exists(f"{LATEX_DIR}/tables"):
    os.makedirs(f"{LATEX_DIR}/tables")


def sigmoid(eta):
    return 1 / (1 + np.exp(-eta))


def get_delta_names(cpd):
    var_names = []
    children_sorted = sorted([k for k in cpd.keys() if "Male" not in k])

    for child in children_sorted:
        parents = cpd[child]["Parents"]
        d = len(parents.keys())
        parents_sorted = sorted(parents.keys())
        if d == 0:
            var_names.append(f"{child} | No parents")
        else:
            var_names += [
                f"{child} | "
                + ", ".join(
                    [
                        f"{pa}={val}"
                        for pa, val in zip(
                            parents_sorted, map(int, reversed(list(f"{j:0{d}b}")))
                        )
                    ]
                )
                for j in range(2**d)
            ]

    return var_names


###############################################################
# Table 3
###############################################################

# Give details on the initial delta
delta_0 = pd.read_csv(f"{DIR}/compare_ipw_taylor_optim/deltas_taylor.csv")
delta_0 = delta_0.drop("Unnamed: 0", axis=1)
delta_0 = delta_0.iloc[0, :]

with open(f"{DIR}/data/train_dist/cpd.pkl", "rb") as f:
    cpd = pickle.load(f)


DELTA_NAME = r"$\delta_i$"

# Print the entire set of conditionals
table_3_df = pd.DataFrame(
    np.concatenate(
        [np.array(get_delta_names(cpd))[:, np.newaxis], delta_0.values[:, np.newaxis]],
        axis=1,
    ),
    columns=["Conditional", DELTA_NAME],
)
table_3_df[DELTA_NAME] = table_3_df[DELTA_NAME].astype(float)
table_3_df["abs"] = np.abs(table_3_df[DELTA_NAME].values)
table_3_sorted = table_3_df.sort_values(by="abs", ascending=False).drop("abs", axis=1)

for a, b in [
    ("=1", "$=1$"),
    ("=0", "$=0$"),
    ("_", " "),
]:
    table_3_sorted["Conditional"] = [
        x.replace(a, b) for x in table_3_sorted["Conditional"]
    ]

# Export to latex
table_3_sorted.round(3).to_latex(
    f"{DIR}/latex/tables/table3.tex", index=False, escape=False
)

###############################################################
# Table 1, Left
###############################################################

top_5 = table_3_sorted[:5]

etas = np.zeros((5))
# WARNING: Manual calculation of the CPD, using the ground-truth statistics
cpd_mustache = cpd["Mustache"]
cpd_smiling = cpd["Smiling"]
cpd_bald = cpd["Bald"]
cpd_lipstick = cpd["Wearing_Lipstick"]
# Mustace | Female, Old
etas[0] = (
    cpd_mustache["Base"]
    + 0 * cpd_mustache["Parents"]["Male"]
    + 0 * cpd_mustache["Parents"]["Young"]
)
# Lipstick | Male, Young
etas[1] = (
    cpd_lipstick["Base"]
    + 1 * cpd_lipstick["Parents"]["Male"]
    + 1 * cpd_lipstick["Parents"]["Young"]
)
# Bald | Male, Young
etas[2] = (
    cpd_bald["Base"]
    + 1 * cpd_bald["Parents"]["Male"]
    + 1 * cpd_bald["Parents"]["Young"]
)
# Lipstick | Male, Old
etas[3] = (
    cpd_lipstick["Base"]
    + 1 * cpd_lipstick["Parents"]["Male"]
    + 0 * cpd_lipstick["Parents"]["Young"]
)
# Smiling | Female, Old
etas[4] = (
    cpd_smiling["Base"]
    + 0 * cpd_smiling["Parents"]["Male"]
    + 0 * cpd_smiling["Parents"]["Young"]
)
top_5[r"$\P$"] = [
    np.round(sigmoid(eta), 3)
    for eta, delta in zip([etas[j] for j in range(5)], top_5[DELTA_NAME])
]
top_5[r"$\P_{\delta}$"] = [
    np.round(sigmoid(eta + delta), 3)
    for eta, delta in zip([etas[j] for j in range(5)], top_5[DELTA_NAME])
]

for a, b in [
    ("Male$=1$", "Male"),
    ("Male$=0$", "Female"),
    ("Young$=0$", "Old"),
    ("Young$=1$", "Young"),
    ("_", " "),
]:
    top_5["Conditional"] = [x.replace(a, b) for x in top_5["Conditional"]]

top_5.round(3).to_latex(
    f"{DIR}/latex/tables/table1_left.tex",
    index=False,
    escape=False,
)

###############################################################
# Table 1, right
###############################################################

df0 = pd.read_csv(f"{DIR}/results/table1_right.csv")
df0.columns = ["Name", "Value"]
df0 = df0.drop(0, axis=0)

rename_maps = {
    "E_taylor actual": r"Ground truth shift acc. ($\E_\delta[\ell_\gamma]$)",
    "Training acc": r"(Original acc ($\E[\ell_\gamma]$))",
    "IPW on Taylor": r"IPW estimate ($\ipw$)",
    "E_taylor": r"Taylor estimate ($\taylor$)",
}

table1_right = df0.set_index("Name").T[rename_maps.keys()].rename(columns=rename_maps)

table1_right.round(3).to_latex(
    f"{DIR}/latex/tables/table1_right.tex",
    index=False,
    escape=False,
)

###############################################################
# Comparison to random shift
###############################################################

RESULTS_PATH = "experiments/celeb_gan/results/"
LOAD_OURS_PATH = "experiments/celeb_gan/compare_ipw_taylor_optim"

df_rand = pd.read_csv(os.path.join(RESULTS_PATH, "figure5_left.csv"))
worst_case_df = pd.read_csv(
    os.path.join(LOAD_OURS_PATH, "results_with_ground_truth_first.csv")
)
worst_case_df.index = worst_case_df["Unnamed: 0"]
number_worse = (df_rand["Loss"] > worst_case_df.loc["E_taylor actual"][1]).mean()

print("EVALUTION AGAINST RANDOM SHIFTS")
print(
    f"\tThe worst-case delta is worse than {100*number_worse:.2f}% of the random deltas"
)

###############################################################
# Overall Results
###############################################################

df = pd.read_csv(f"{DIR}/compare_ipw_taylor_optim/results_with_ground_truth.csv")

print("EVALUATION OF IMPACT OF TAYLOR")
# General drop in accuracy from original to shifted distribution
training_acc = df["Training acc"]
print(f"\tOriginal Accuracy: {np.round(training_acc.mean(), 4)}")

acc_drop = df["Training acc"] - df["E_taylor actual"]
print(
    (
        f"\tDrop in Accuracy: {np.round(acc_drop.mean(), 4)}\n"
        f"\t\t with Std Dev {np.round(acc_drop.std(), 4)}"
    )
)

print("EVALUATION OF RUNTIME")
# Runtime difference
runtime_ipw = df["elapsed_ipw"]
runtime_taylor = df["elapsed_taylor"]
print(
    (
        f"\tRuntime IPW: {np.round(runtime_ipw.mean(), 4)}\n"
        f"\tRuntime Taylor: {np.round(runtime_taylor.std(), 4)}"
    )
)

print("EVALUATION OF EFFECTIVENESS")
# Average difference in effectiveness
actual_ipw = df["E_ipw actual"]
actual_taylor = df["E_taylor actual"]
print(
    (
        f"\tShifted Acc IPW: {np.round(actual_ipw.mean(), 4)}\n"
        f"\tShifted Acc Taylor: {np.round(actual_taylor.mean(), 4)}\n"
        f"\tDifference (mean): {np.round((actual_taylor - actual_ipw).mean(), 4)}\n"
        f"\tDifference (std): {np.round((actual_taylor - actual_ipw).std(), 4)}\n"
        f"\t% Taylor is Better: {np.round((actual_taylor - actual_ipw < 0).mean(), 2)}"
    )
)

print("EVALUATION OF ACCURACY ON CHOSEN SHIFTS")
# Differences in accuracy (for estimating one's own shift)
est_ipw = df["E_ipw"]
est_taylor = df["E_taylor"]
actual_ipw = df["E_ipw actual"]
actual_taylor = df["E_taylor actual"]
print(
    (
        f"\tMAE IPW: {np.round((est_ipw - actual_ipw).abs().mean(), 4)}\n"
        f"\tMAE Taylor: {np.round((est_taylor - actual_taylor).abs().mean(), 4)}"
    )
)

print("EVALUATION OF ACCURACY ON TAYLOR SHIFTS")
est_ipw_on_taylor = df["IPW on Taylor"]
est_taylor = df["E_taylor"]
actual_taylor = df["E_taylor actual"]
rmse_ipw = np.sqrt(np.mean(np.square(est_ipw_on_taylor - actual_taylor)))
rmse_taylor = np.sqrt(np.mean(np.square(est_taylor - actual_taylor)))
print(
    (
        f"\tRMSE IPW: {np.round(rmse_ipw, 4)}\n"
        f"\tRMSE Taylor: {np.round(rmse_taylor, 4)}\n"
    )
)
