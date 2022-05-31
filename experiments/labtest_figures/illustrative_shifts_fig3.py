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

SMALL_SIZE = 14
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

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
# Given new data, make predictions, calculate loss
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


def loss_func(y_true, y_pred):
    return sklearn.metrics.log_loss(y_true, y_pred)


def loss_func_unit(y_true, y_pred):
    return -1 * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def est_quad_1d(data, preds, delta):
    data['preds'] = preds
    data['loss'] = loss_func_unit(data['Y'], data['preds'])
    train_loss = data['loss'].mean()

    # Manually compute the shift gradient
    data_y0 = data.query('Y == 0')
    means_y0 = data_y0.mean()
    cond_cov_y0 = ((data_y0['loss'] - means_y0['loss']) *
                   (data_y0['O'] - means_y0['O'])).mean()
    data_y1 = data.query('Y == 1')
    means_y1 = data_y1.mean()
    cond_cov_y1 = ((data_y1['loss'] - means_y1['loss']) *
                   (data_y1['O'] - means_y1['O'])).mean()

    py = data['Y'].mean()
    sgrad = cond_cov_y0 * (1 - py) + cond_cov_y1 * p_y

    # Manually compute the shift hessian
    exp_OcY = means_y1['O'] * data['Y'] + means_y0['O'] * (1 - data['Y'])
    data['O_eps'] = np.square(data['O'] - exp_OcY)

    data_y0 = data.query('Y == 0')
    means_y0 = data_y0.mean()
    cond_cov_eps_y0 = ((data_y0['loss'] - means_y0['loss']) *
                       (data_y0['O_eps'] - means_y0['O_eps'])).mean()
    data_y1 = data.query('Y == 1')
    means_y1 = data_y1.mean()
    cond_cov_eps_y1 = ((data_y1['loss'] - means_y1['loss']) *
                       (data_y1['O_eps'] - means_y1['O_eps'])).mean()

    shess = cond_cov_eps_y0 * (1 - py) + cond_cov_eps_y1 * p_y

    return train_loss + delta * sgrad + 0.5 * delta**2 * shess


o_losses = []
lo_losses = []
est_quad_losses = []
none_losses = []
testing_rates = []

delta_space = np.linspace(-5, 5, num=50)

val_data = generate_data(alpha=alpha, beta=beta, n=100000, seed=1)
pred_o = cls_o.predict_proba(val_data[['O']])[:, 1]
pred_l = cls_lo.predict_proba(val_data[['L']])[:, 1]
pred_lo_val = pred_o * (1 - val_data['O']) + pred_l * val_data['O']

max_est_quad = (0, -100)
delta_bound = 2

for delta in delta_space:
    test_data = generate_data(alpha=alpha + delta, beta=beta, n=100000, seed=1)
    pred_o = cls_o.predict_proba(test_data[['O']])[:, 1]
    pred_l = cls_lo.predict_proba(test_data[['L']])[:, 1]
    pred_lo = pred_o * (1 - test_data['O']) + pred_l * test_data['O']
    pred_none = np.ones_like(pred_o) * mean_y
    testing_rate = test_data['O'].mean()
    testing_rates.append(testing_rate)

    o_losses.append(sklearn.metrics.log_loss(test_data['Y'], pred_o))
    lo_losses.append(sklearn.metrics.log_loss(test_data['Y'], pred_lo))
    none_losses.append(sklearn.metrics.log_loss(test_data['Y'], pred_none))

    est_loss = est_quad_1d(val_data, pred_lo_val, delta)
    est_quad_losses.append(est_loss)
    if max_est_quad[1] < est_loss and abs(delta) < delta_bound:
        max_est_quad = (delta, est_loss)

loss_df = pd.DataFrame(
    data={
        '$\delta_0$': delta_space,
        '$P_{\delta_0}(O = 1)$': testing_rates,
        'Loss of f(O)': o_losses,
        'Loss of f(O, L)': lo_losses,
        'Loss of Base Rate': none_losses,
        'Est Loss': est_quad_losses,
    })

orig_plot_df = pd.melt(loss_df,
                       id_vars=['$\delta_0$', '$P_{\delta_0}(O = 1)$'],
                       var_name='Model',
                       value_name='Loss')

# Only plot the last model
plot_names = ['Loss of f(O, L)', 'Est Loss']
plot_df = orig_plot_df.query("Model in @plot_names")

ymin = 0.5
ymax = 0.85
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
sns.lineplot(data=plot_df, x='$\delta_0$', y='Loss', hue='Model', ax=ax)
ax.axvline(x=0, label='Training Dist.', color='k', linestyle='dotted')
plt.fill_betweenx(np.linspace(ymin, ymax),
                  x1=-delta_bound,
                  x2=delta_bound,
                  label='Robustness Set',
                  color='k',
                  alpha=0.2)
ax.set_ylim(ymin, ymax)
ax.plot(max_est_quad[0],
        max_est_quad[1],
        linestyle='None',
        color='r',
        marker='*',
        label='Est. Worst-Case')
ax.legend(loc='upper right')
# leg = ax.legend(bbox_to_anchor=(-0.25, 1))
# leg.set_in_layout(False)
plt.tight_layout()
plt.savefig(f'{fpath}/labtest_delta_shift_onlyOL_quad_est.pdf')

# Only plot the last model
plot_names = ['Loss of f(O, L)']
plot_df = orig_plot_df.query("Model in @plot_names")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
sns.lineplot(data=plot_df,
             x='$P_{\delta_0}(O = 1)$',
             y='Loss',
             hue='Model',
             ax=ax)
ax.set_ylim(ymin, ymax)
ax.axvline(x=0.5, label='Training Dist.', color='k', linestyle='dotted')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'{fpath}/labtest_labtest_shift_onlyOL.pdf')
