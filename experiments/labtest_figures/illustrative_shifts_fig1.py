#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple script to simulate different types of shift."""

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import pandas as pd  # type: ignore
import sklearn  # type: ignore
import itertools
from sklearn import linear_model as lm  # type: ignore
from mpl_toolkits import mplot3d

SMALL_SIZE = 20
MEDIUM_SIZE = 30
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

colors = sns.color_palette('colorblind')

fpath = './figs'


def sigmoid(x):
    """Return Sigmoid Function."""
    return 1 / (1 + np.exp(-x))


class DataGenerator(object):
    """Docstring for DataGenerator. """

    def __init__(self, seed=0, n=100):
        self.default_seed = seed
        self.default_n = n
        self.disease = None
        self.age = None
        # self.symptoms = None
        self.order = None
        self.test_result = None

    def _simulate_age(self):
        mean = 0
        std = 0.5
        self.age = self.rng.normal(loc=mean, scale=std, size=self.n)

    def _simulate_disease(self):
        assert self.age is not None, 'Age not simulated yet!'
        gamma = 0.5
        prob_y = sigmoid(self.age * gamma - gamma / 2)
        self.disease = self.rng.binomial(1, prob_y)

    # def _simulate_symptoms(self):
    #     for var in [self.age, self.disease]:
    #         assert var is not None
    #
    #     intercept = -1
    #     coef_age = 0.5
    #     coef_disease = 0.5
    #     prob_symptoms = sigmoid(intercept + self.disease * coef_disease +
    #                             self.age * coef_age)
    #     self.symptoms = self.rng.binomial(1, prob_symptoms)

    def _simulate_order(self, dy0=0, dy1=0):
        for var in [self.age, self.disease]:
            assert var is not None

        intercept = -1
        coef_age = 0.5
        coef_disease = 2
        # coef_symptom = 0.5

        shift = dy0 * (1 - self.disease) + dy1 * self.disease

        prob_order = sigmoid(intercept + self.disease * coef_disease +
                             self.age * coef_age +
                             # self.symptoms * coef_symptom +
                             shift)
        self.order = self.rng.binomial(1, prob_order)

    def _simulate_test_result(self):
        parents = [self.age, self.disease, self.order]
        for var in parents:
            assert var is not None

        mu = 0.5
        sigma = 1
        cond_mean = (2 * self.disease - 1) * mu

        self.test_result = self.rng.normal(loc=cond_mean, scale=sigma)
        self.test_result[self.order == 0] = 0

    def _reset(self, seed=None, n=None):
        if seed is None:
            seed = self.default_seed
        if n is None:
            n = self.default_n

        self.rng = np.random.default_rng(seed)
        self.n = n

    def _simulate_all(self, dy0_order=0, dy1_order=0):
        self._simulate_age()
        self._simulate_disease()
        # self._simulate_symptoms()
        self._simulate_order(dy0=dy0_order, dy1=dy1_order)
        self._simulate_test_result()

    def _return_data(self):
        self.data = {
            'Age': self.age,
            'Disease': self.disease,
            # 'Symptoms': self.symptoms,
            'Order': self.order,
            'TestResult': self.test_result
        }
        return pd.DataFrame(data=self.data)

    def generate_data(self, seed=None, n=None, dy0_order=0, dy1_order=0):
        self._reset(seed, n)
        self._simulate_all(dy0_order=dy0_order, dy1_order=dy1_order)
        return self._return_data()


def select_Xy(data, order=True, subset=True):
    label = 'Disease'
    if order:
        features = ['Age', 'Order', 'TestResult']
    else:
        features = ['Age', 'Order']
    # if order:
    #     features = ['Age', 'Symptoms', 'Order', 'TestResult']
    # else:
    #     features = ['Age', 'Symptoms', 'Order']

    if subset:
        data = data.query('Order == @order')

    return data[features], data[label]


def train_components(data):
    # Get training data
    X_no_order, Y_no_order = select_Xy(data, order=False, subset=True)
    X_order, Y_order = select_Xy(data, order=True, subset=True)

    # Initialize and fit classifiers
    cls_no_order = lm.LogisticRegression(penalty='none')
    cls_order = lm.LogisticRegression(penalty='none')
    cls_no_order.fit(X_no_order, Y_no_order)
    cls_order.fit(X_order, Y_order)

    return {'no_order': cls_no_order, 'order': cls_order}


def predict_proba(data, model_d):
    assert 'no_order' in model_d.keys()
    assert 'order' in model_d.keys()

    # Get predictions from both models for all samples
    X_no_order, Y_no_order = select_Xy(data, order=False, subset=False)
    X_order, Y_order = select_Xy(data, order=True, subset=False)
    assert np.array_equal(Y_no_order, Y_order)

    pred_order = model_d['order'].predict_proba(X_order)[:, 1]
    pred_no_order = model_d['no_order'].predict_proba(X_no_order)[:, 1]

    return pred_order * data['Order'] + pred_no_order * (1 - data['Order'])


def loss_func_unit(y_true, y_pred):
    return -1 * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


########################################################################
## Train the model
########################################################################

# Generate data and fit a model
gen = DataGenerator(seed=0, n=100000)
train_data = gen.generate_data(dy0_order=0, dy1_order=0)
label = 'Disease'
model_d = train_components(train_data)

# Summary statistics
print(f'Testing rate: {train_data["Order"].mean()}')

##############################################################################
# 3-dimensional plotting !!!!
##############################################################################

len_grid = 100
delta_space = np.linspace(-5, 5, num=len_grid)
X, Y = np.meshgrid(delta_space, delta_space)
Z = np.zeros((len_grid, len_grid))

# Example: plot the actual loss at each value of delta
for i in range(len_grid):
    for j in range(len_grid):
        # This implements "simulate the loss for this shift
        test_data = gen.generate_data(dy0_order=X[i, j],
                                      dy1_order=Y[i, j],
                                      n=10000,
                                      seed=1)

        preds = predict_proba(test_data, model_d)
        loss = np.mean(loss_func_unit(test_data[label], preds))
        Z[i, j] = loss

# Get loss at origin
test_data = gen.generate_data(dy0_order=0, dy1_order=0, n=10000, seed=1)
preds = predict_proba(test_data, model_d)
origin_loss = np.mean(loss_func_unit(test_data[label], preds))

from matplotlib import cm

fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": "3d"})
ax.plot_surface(Y, X, Z, cmap=cm.coolwarm, alpha=0.9)
ax.plot(
    [0, 0],
    [0, 0],
    [origin_loss, origin_loss],  # origin_loss, 1],
    color='k',
    zorder=10,
    marker='*',
    markersize=15,
    linestyle='None',
    label='Training Distribution')
ax.invert_xaxis()
ax.set_ylabel('\n\n$\delta_0$')
ax.set_xlabel('\n\n$\delta_1$')
ax.set_zlabel('\n\nLoss under $(\delta_0, \delta_1)$')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
plt.legend().set_visible(False)
plt.tight_layout()
plt.savefig('figs/3d_figure1.pdf', bbox_inches='tight')
