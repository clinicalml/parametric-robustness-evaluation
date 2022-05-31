#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple script to simulate different types of shift."""

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import pandas as pd  # type: ignore
import sklearn  # type: ignore
from sklearn import linear_model as lm  # type: ignore
from matplotlib.legend_handler import HandlerPatch  # type: ignore
import matplotlib.patches as mpatches  # type: ignore

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height,
                      fontsize):
    p = mpatches.FancyArrow(0,
                            0.5 * height,
                            width,
                            0,
                            length_includes_head=True,
                            head_width=0.75 * height)
    return p


handler_map = {
    mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow),
}

colors = sns.color_palette('colorblind')

fpath = './figs'


def sigmoid(x):
    """Return Sigmoid Function."""
    return 1 / (1 + np.exp(-x))


# Parameters
p_y = 0.5
alpha = -1
beta = 2
mu = 0.5
sigma = 1


# Generate Tables of different probabilities
def cp_order_disease(y, a=alpha, b=beta):
    """Return conditional probability of ordering a test."""
    return sigmoid(a + b * y)


def mp_order(a=alpha, b=beta):
    """Return marginal probability of ordering a test."""
    return p_y * cp_order_disease(1, a, b) + (1 - p_y) * cp_order_disease(
        0, a, b)


def cp_disease_order(o, a=alpha, b=beta):
    """Return conditional probability of disease, given test order."""
    # Bayes rule: P(Y | O) = P(O | Y) P(Y) / P(O)
    po = mp_order(a=a, b=b) * o + (1 - mp_order(a=a, b=b)) * (1 - o)
    py = p_y
    poy = cp_order_disease(
        y=1, a=a, b=b) * o + (1 - cp_order_disease(y=1, a=a, b=b)) * (1 - o)
    return poy * py / po


def generate_data(alpha: float, beta: float, n: int,
                  seed: int) -> pd.DataFrame:
    """Generate synthetic data."""
    rng = np.random.default_rng(seed=seed)
    disease = rng.binomial(1, p_y, size=n)
    order = rng.binomial(1, cp_order_disease(disease, alpha, beta))
    labvalue = rng.normal(loc=(2 * disease - 1) * mu, scale=sigma)
    labvalue[order == 0] = 0

    return pd.DataFrame(data={'Y': disease, 'O': order, 'L': labvalue})


# ############################################################################
# Figure: Show the worst-case shift direction in 2D, as well as the worst-case
# shift under the DRO constraint
# ############################################################################

train_data = generate_data(alpha=alpha, beta=beta, n=100000, seed=0)

cls_o = lm.LogisticRegression(penalty='none')
cls_o.fit(train_data[['O']], train_data['Y'])

cls_lo = lm.LogisticRegression(penalty='none')
cls_lo.fit(train_data.query('O == 1')[['L']], train_data.query('O == 1')['Y'])

mean_y = train_data['Y'].mean()

# Get the validation loss
val_data = generate_data(alpha=alpha, beta=beta, n=100000, seed=1)
pred_o = cls_o.predict_proba(val_data[['O']])[:, 1]
pred_l = cls_lo.predict_proba(val_data[['L']])[:, 1]
pred_lo = pred_o * (1 - val_data['O']) + pred_l * val_data['O']
pred_none = np.ones_like(pred_o) * mean_y

loss_d = {
    'lo': {
        'name': 'f(O, L)',
        'color': colors[1],
        'pred': pred_lo
    },
}

loss_func = sklearn.metrics.log_loss

for k, v in loss_d.items():
    this_data = val_data.copy()
    this_data['pred'] = v['pred']

    this_data_11 = this_data.query('O == 1 & Y == 1')
    this_data_10 = this_data.query('O == 1 & Y == 0')
    this_data_01 = this_data.query('O == 0 & Y == 1')
    this_data_00 = this_data.query('O == 0 & Y == 0')

    v['cond_loss'] = {
        'mu11':
            loss_func(this_data_11['Y'], this_data_11['pred'], labels=[0, 1]),
        'mu10':
            loss_func(this_data_10['Y'], this_data_10['pred'], labels=[0, 1]),
        'mu01':
            loss_func(this_data_01['Y'], this_data_01['pred'], labels=[0, 1]),
        'mu00':
            loss_func(this_data_00['Y'], this_data_00['pred'], labels=[0, 1]),
    }

    v['worst_dir'] = {
        'O1cY1': p_y * (v['cond_loss']['mu11'] - v['cond_loss']['mu01']),
        'O1cY0': (1 - p_y) * (v['cond_loss']['mu10'] - v['cond_loss']['mu00'])
    }

    # Assume, as in our plots, that x = O1cY0 and y = O1cY1
    vec = np.array([v['worst_dir']['O1cY0'], v['worst_dir']['O1cY1']])
    v['worst_dir']['vec'] = vec

    # L2 norm vector
    v['worst_dir']['unit_vec'] = vec / np.linalg.norm(vec)

labels = {
    'cp_healthy': '$P(O = 1 \mid Y = 0)$',
    'cp_unhealthy': '$P(O = 1 \mid Y = 1)$',
}

# Plot the worst-case direction, and the worst-case value
for shift_alpha in [0.4, 0.6, 0.8]:
    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=(5, 5),
                           sharex=True,
                           sharey=True)

    p11 = cp_order_disease(y=1)
    p10 = cp_order_disease(y=0)

    p11_lower_bound = np.max([0, (p11 - shift_alpha) / (1 - shift_alpha)])
    p11_upper_bound = np.min([1, p11 / (1 - shift_alpha)])

    p10_lower_bound = np.max([0, (p10 - shift_alpha) / (1 - shift_alpha)])
    p10_upper_bound = np.min([1, p10 / (1 - shift_alpha)])

    new_p10_range = np.linspace(p10_lower_bound, p10_upper_bound)

    # Get the worst-case choice for each model under the 1-alpha constraint
    for k, v in loss_d.items():
        if v['cond_loss']['mu11'] > v['cond_loss']['mu01']:
            v['worst_q11'] = p11_upper_bound
        else:
            v['worst_q11'] = p11_lower_bound

        if v['cond_loss']['mu10'] > v['cond_loss']['mu00']:
            v['worst_q10'] = p10_upper_bound
        else:
            v['worst_q10'] = p10_lower_bound

    # Plot the feasible region
    plt.fill_between(new_p10_range,
                     p11_lower_bound,
                     p11_upper_bound,
                     color='k',
                     alpha=0.2,
                     label='Robustness Set')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel(labels['cp_healthy'])
    ax.set_ylabel(labels['cp_unhealthy'])

    # Plot the original point
    ax.plot(cp_order_disease(y=0),
            cp_order_disease(y=1),
            color='k',
            marker='o',
            markersize=10,
            linestyle='None',
            label='Original Distribution')

    # Plot the worst-case direction, starting at the origin
    for k, v in loss_d.items():
        plt.arrow(
            cp_order_disease(0),
            cp_order_disease(1),
            v['worst_dir']['unit_vec'][0] * 0.2,
            v['worst_dir']['unit_vec'][1] * 0.2,
            color='red',
            # color=v['color'],
            label='Worst-Case Direction',
            # label=v['name'],
            head_width=0.02,
            alpha=0.2)

        ax.plot(v['worst_q10'],
                v['worst_q11'],
                color='red',
                marker='*',
                markersize=10,
                linestyle='None',
                label='Worst-Case Distribution')

    if shift_alpha == 0.8:
        plt.legend(numpoints=1, handler_map=handler_map)
    else:
        plt.legend().set_visible(False)

    plt.savefig(
        f'{fpath}/labtest_subpopulation_worst_case_alpha{shift_alpha}.pdf')

# ############################################################################
# Figure: Show the worst-case shift direction in 2D, as well as the worst-case
# shift under OUR constraints.
# ############################################################################

train_data = generate_data(alpha=alpha, beta=beta, n=100000, seed=0)

cls_o = lm.LogisticRegression(penalty='none')
cls_o.fit(train_data[['O']], train_data['Y'])

cls_lo = lm.LogisticRegression(penalty='none')
cls_lo.fit(train_data.query('O == 1')[['L']], train_data.query('O == 1')['Y'])

mean_y = train_data['Y'].mean()

# Get the validation loss
val_data = generate_data(alpha=alpha, beta=beta, n=100000, seed=1)
pred_o = cls_o.predict_proba(val_data[['O']])[:, 1]
pred_l = cls_lo.predict_proba(val_data[['L']])[:, 1]
pred_lo = pred_o * (1 - val_data['O']) + pred_l * val_data['O']
pred_none = np.ones_like(pred_o) * mean_y

loss_d = {
    'lo': {
        'name': 'L, O',
        'color': colors[1],
        'pred': pred_lo
    },
}


def loss_func(y_true, y_pred):
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
    return sklearn.metrics.zero_one_loss(y_true, y_pred)
    # return sklearn.metrics.log_loss(y_true, y_pred, labels=[0, 1])


# Pick a model to evaluate here
model = 'lo'
validation_loss = loss_func(this_data['Y'], loss_d[model]['pred'])
print(f'Model ({model}) with validation loss of {validation_loss:.2f}')

# Get mu(i, j) for each model
for k, v in loss_d.items():
    this_data = val_data.copy()
    this_data['pred'] = v['pred']

    this_data_11 = this_data.query('O == 1 & Y == 1')
    this_data_10 = this_data.query('O == 1 & Y == 0')
    this_data_01 = this_data.query('O == 0 & Y == 1')
    this_data_00 = this_data.query('O == 0 & Y == 0')

    v['cond_loss'] = {
        'mu11': loss_func(this_data_11['Y'], this_data_11['pred']),
        'mu10': loss_func(this_data_10['Y'], this_data_10['pred']),
        'mu01': loss_func(this_data_01['Y'], this_data_01['pred']),
        'mu00': loss_func(this_data_00['Y'], this_data_00['pred']),
    }

    v['worst_dir'] = {
        'O1cY1': p_y * (v['cond_loss']['mu11'] - v['cond_loss']['mu01']),
        'O1cY0': (1 - p_y) * (v['cond_loss']['mu10'] - v['cond_loss']['mu00'])
    }

    # Assume, as in our plots, that x = O1cY0 and y = O1cY1
    vec = np.array([v['worst_dir']['O1cY0'], v['worst_dir']['O1cY1']])
    v['worst_dir']['vec'] = vec

    # L2 norm vector
    v['worst_dir']['unit_vec'] = vec / np.linalg.norm(vec)

# [To Do]: Make this into a sequence of figures, to find plausible worst-case
# sets, and compute the resulting loss under these sets.

labels = {
    'cp_healthy': '$P(O = 1 \mid Y = 0)$',
    'cp_unhealthy': '$P(O = 1 \mid Y = 1)$',
    'delta_a': '$\\delta_{0}$',
    'delta_b': '$\\delta_{1}$',
}

delta_lims = [(-10, 10, -20, 20), (-5, 5, -1, 1), (-1.05, 1.05, 0, 0)]

for i in range(len(delta_lims)):
    delta_a_min, delta_a_max, delta_b_min, delta_b_max = delta_lims[i]
    delta_a_space = np.linspace(delta_a_min, delta_a_max, 41)
    delta_b_space = np.linspace(delta_b_min, delta_b_max, 11)
    assert 0 in delta_b_space
    assert 0 in delta_a_space

    print(f'User Story {i}')
    print(f'\tDelta0 in [{delta_a_min}, {delta_a_max}]')
    print(f'\tDelta1 in [{delta_b_min}, {delta_b_max}]')

    plot_data = pd.DataFrame([{
        labels['cp_healthy']:
            cp_order_disease(y=0, a=alpha + delta_a, b=beta + delta_b),
        labels['cp_unhealthy']:
            cp_order_disease(y=1, a=alpha + delta_a, b=beta + delta_b),
        labels['delta_a']:
            delta_a,
        labels['delta_b']:
            delta_b
    } for delta_a in delta_a_space for delta_b in delta_b_space])

    # for model, d in enumerate(loss_d):

    d = loss_d[model]['cond_loss']

    loss_df = plot_data.copy()
    loss_df['pred_loss'] = (
        (p_y * (d['mu11'] * loss_df[labels['cp_unhealthy']] + d['mu01'] *
                (1 - loss_df[labels['cp_unhealthy']]))) +
        ((1 - p_y) * (d['mu10'] * loss_df[labels['cp_healthy']] + d['mu00'] *
                      (1 - loss_df[labels['cp_healthy']]))))

    worst_case_x = loss_df.sort_values(
        by='pred_loss',
        ascending=False).head(1)[labels['cp_healthy']].values[0]
    worst_case_y = loss_df.sort_values(
        by='pred_loss',
        ascending=False).head(1)[labels['cp_unhealthy']].values[0]
    worst_case_loss = loss_df.sort_values(
        by='pred_loss', ascending=False).head(1)['pred_loss'].values[0]

    print(f'\tWorst case (p10, p11): [{worst_case_x:.2f}, {worst_case_y:.2f}]')
    print(f'\tEstimated Worst Case Loss: {worst_case_loss:.2f}')

    # Plot the bounds
    x_vals = plot_data[plot_data[labels['delta_b']] == 0][labels['cp_healthy']]
    y_lower = plot_data[plot_data[labels['delta_b']] == delta_b_max][
        labels['cp_unhealthy']]
    y_upper = plot_data[plot_data[labels['delta_b']] == delta_b_min][
        labels['cp_unhealthy']]

    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=(5, 5),
                           sharex=True,
                           sharey=True)

    # Check that we're getting the right values
    # sns.scatterplot(x=plot_data[labels['cp_healthy']],
    #                 y=plot_data[labels['cp_unhealthy']])

    plt.fill_between(x_vals,
                     y_lower,
                     y_upper,
                     alpha=0.2,
                     color='k',
                     label='Robustness Set')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel(labels['cp_healthy'])
    ax.set_ylabel(labels['cp_unhealthy'])
    # Plot the original distribution
    ax.plot(cp_order_disease(y=0, a=alpha),
            cp_order_disease(y=1, a=alpha),
            color='k',
            marker='o',
            markersize=10,
            label='Original Distribution',
            linestyle='None')

    ax.plot(worst_case_x,
            worst_case_y,
            color='r',
            marker='*',
            markersize=10,
            label='Worst-Case Distribution',
            linestyle='None')

    # Plot the worst-case direction
    v = loss_d[model]
    plt.arrow(cp_order_disease(0, a=alpha),
              cp_order_disease(1, a=alpha),
              v['worst_dir']['unit_vec'][0] * 0.2,
              v['worst_dir']['unit_vec'][1] * 0.2,
              color='r',
              label='Worst-Case Direction',
              head_width=0.02,
              alpha=0.2)

    if i == 0:
        plt.legend(numpoints=1, handler_map=handler_map)
    else:
        plt.legend().set_visible(False)

    # Plot the worst-case distribution
    plt.savefig(f'{fpath}/labtest_user_story_{i}.pdf')
