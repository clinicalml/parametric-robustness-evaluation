#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple script to simulate different types of shift."""

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import pandas as pd  # type: ignore
from sklearn import linear_model as lm  # type: ignore
from tqdm import tqdm 
import os

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
n_reps  = 200

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Save data as csvs
SAVE_PATH = "experiments/labtest_ipw_taylor_comparison/plot"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


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

    def _simulate_age(self, mean=0, std=0.5):
        mean = mean
        std = std
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

    def _simulate_all(self, dy0_order=0, dy1_order=0, mean_age=0, std_age=0.5):
        self._simulate_age(mean=mean_age, std=std_age)
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

    def generate_data(self, seed=None, n=None, dy0_order=0, dy1_order=0, mean_age=0, std_age=0.5):
        self._reset(seed, n)
        self._simulate_all(dy0_order=dy0_order, dy1_order=dy1_order, mean_age=mean_age, std_age=std_age)
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


def weights(data, dy0, dy1):
    disease = data['Disease']
    age = data['Age']
    order = data['Order']
    intercept = -1
    coef_age = 0.5
    coef_disease = 2

    shift = dy0 * (1 - disease) + dy1 * disease

    p_original = sigmoid(intercept + disease * coef_disease + age * coef_age)
    p_shift = sigmoid(intercept + disease * coef_disease + age * coef_age +
                      shift)

    weight_o1 = p_shift / p_original
    weight_o0 = (1 - p_shift) / (1 - p_original)

    return weight_o1 * order + weight_o0 * (1 - order)


def get_true_proba(data):
    disease = data['Disease']
    age = data['Age']
    intercept = -1
    coef_age = 0.5
    coef_disease = 2
    return sigmoid(intercept + disease * coef_disease + age * coef_age)


def get_taylor_estimate(data, loss, dy0, dy1):
    p = get_true_proba(data)
    ordered = data['Order'].values

    D_1 = np.array([(ordered == 0), (ordered == 1)])
    D_1_squared = np.stack([np.diag([o, 1 - o]) for o in ordered]).T

    sg1 = (np.array(loss * (ordered - p)) * D_1).T.mean(axis=0).reshape(-1, 1)
    sg2 = (D_1_squared * np.array(loss * ((ordered - p)**2 -
                                          (p - p**2)))).mean(axis=2)
    delta = np.array([dy0, dy1]).reshape(-1, 1)

    return (loss.mean() + delta.T @ sg1 + 1 / 2 * delta.T @ sg2 @ delta)[0, 0]


# Summary statistics
print(f'Testing rate: {train_data["Order"].mean()}')

########################################################################
## Plot the 1D loss curve
########################################################################

# for seed in range(10):
seed = 1
val_datas = [
    gen.generate_data(dy0_order=0, dy1_order=0, seed=seed_, n=1000)
    for seed_ in range(1, 1+n_reps)
]
val_preds = [predict_proba(val_data, model_d) for val_data in val_datas]
val_losses = [
    loss_func_unit(val_data[label], val_pred)
    for val_data, val_pred in zip(val_datas, val_preds)
]

# Validate across settings
true_loss = []
ipw_losses_lower = []
ipw_losses_upper = []
ipw_losses_median = []
ipw_losses_mean = []
ipw_losses_std = []

taylor_losses_lower = []
taylor_losses_upper = []
taylor_losses_median = []
taylor_losses_mean = []
taylor_losses_std = []

testing_rates = []
delta_space = np.linspace(-3, 3, num=100)

df_to_R = []

for delta in tqdm(delta_space):
    # Estimated IPW losses
    est_ipw_loss = [
        np.mean(val_loss * weights(val_data, dy0=delta, dy1=delta))
        for (val_loss, val_data) in zip(val_losses, val_datas)
    ]

    ipw_losses_lower.append(np.quantile(est_ipw_loss, 0.05))
    ipw_losses_upper.append(np.quantile(est_ipw_loss, 0.95))
    ipw_losses_median.append(np.median(est_ipw_loss))
    ipw_losses_mean.append(np.mean(est_ipw_loss))
    ipw_losses_std.append(np.std(est_ipw_loss))

    # Estimated Taylor losses
    est_taylor_losses = [
        get_taylor_estimate(val_data, val_loss, dy0=delta, dy1=delta)
        for (val_loss, val_data) in zip(val_losses, val_datas)
    ]
    taylor_losses_lower.append(np.quantile(est_taylor_losses, 0.05))
    taylor_losses_upper.append(np.quantile(est_taylor_losses, 0.95))
    taylor_losses_median.append(np.median(est_taylor_losses))
    taylor_losses_mean.append(np.mean(est_taylor_losses))
    taylor_losses_std.append(np.std(est_taylor_losses))

    # True losses under the test distribution
    test_data = gen.generate_data(dy0_order=delta,
                                  dy1_order=delta,
                                  n=100000,
                                  seed=1)
    testing_rate = test_data['Order'].mean()
    testing_rates.append(testing_rate)

    preds = predict_proba(test_data, model_d)
    loss = np.mean(loss_func_unit(test_data[label], preds))
    true_loss.append(loss)

    df_to_R.append({"delta": delta, "std": np.std(est_taylor_losses), "mean": np.mean(est_taylor_losses), "variable": "order", "method": "taylor", "lower": np.quantile(est_taylor_losses, 0.05), "upper": np.quantile(est_taylor_losses, 0.95), "median": np.median(est_taylor_losses)})
    df_to_R.append({"delta": delta, "std": np.std(est_ipw_loss), "mean": np.mean(est_ipw_loss), "variable": "order", "method": "ipw", "lower": np.quantile(est_ipw_loss, 0.05), "upper": np.quantile(est_ipw_loss, 0.95), "median": np.median(est_ipw_loss)})
    df_to_R.append({"delta": delta, "std": 0, "mean": loss, "variable": "order", "method": "true", "lower": loss, "upper": loss})


#==============================================================================
# Gaussian
#==============================================================================
from scipy.stats import norm
def weights(data, delta):
    age = data['Age']
    mean_age = 0
    std_age = 0.5

    p_original = norm(loc=mean_age, scale=std_age).pdf(age)
    p_shift = norm(loc=mean_age + delta, scale=std_age).pdf(age)

    return p_shift/p_original


def get_taylor_estimate(data, loss, delta):
    age = data['Age'].values
    std = 0.5


    sg1 = np.array(loss * (age/std)).mean()
    sg2 = np.array(loss * ((age/std)**2 - (std/std)**2)).mean()

    return (loss.mean() + delta*sg1 + 1/2 * delta**2*sg2)

# for seed in range(10):
seed = 1
val_datas = [
    gen.generate_data(seed=seed_, n=1000)
    for seed_ in range(1, 1+n_reps)
]
val_preds = [predict_proba(val_data, model_d) for val_data in val_datas]
val_losses = [
    loss_func_unit(val_data[label], val_pred)
    for val_data, val_pred in zip(val_datas, val_preds)
]

# Validate across settings
true_loss = []
ipw_losses_lower = []
ipw_losses_upper = []
ipw_losses_median = []
ipw_losses_mean = []
ipw_losses_std = []

taylor_losses_lower = []
taylor_losses_upper = []
taylor_losses_median = []
taylor_losses_mean = []
taylor_losses_std = []

testing_rates = []
delta_space = np.linspace(-3, 3, num=100)

for delta in tqdm(delta_space):
    # Estimated IPW losses
    est_ipw_loss = [
        np.mean(val_loss * weights(val_data, delta=delta))
        for (val_loss, val_data) in zip(val_losses, val_datas)
    ]

    ipw_losses_lower.append(np.quantile(est_ipw_loss, 0.05))
    ipw_losses_upper.append(np.quantile(est_ipw_loss, 0.95))
    ipw_losses_median.append(np.median(est_ipw_loss))
    ipw_losses_mean.append(np.mean(est_ipw_loss))
    ipw_losses_std.append(np.std(est_ipw_loss))

    # Estimated Taylor losses
    est_taylor_losses = [
        get_taylor_estimate(val_data, val_loss, delta=delta)
        for (val_loss, val_data) in zip(val_losses, val_datas)
    ]

    taylor_losses_lower.append(np.quantile(est_taylor_losses, 0.05))
    taylor_losses_upper.append(np.quantile(est_taylor_losses, 0.95))
    taylor_losses_median.append(np.median(est_taylor_losses))
    taylor_losses_mean.append(np.mean(est_taylor_losses))
    taylor_losses_std.append(np.std(est_taylor_losses))

    # True losses under the test distribution
    test_data = gen.generate_data(mean_age=delta,
                                  n=100000,
                                  seed=1)
    testing_rate = test_data['Order'].mean()
    testing_rates.append(testing_rate)

    preds = predict_proba(test_data, model_d)
    loss = np.mean(loss_func_unit(test_data[label], preds))
    true_loss.append(loss)

    df_to_R.append({"delta": delta, "std": np.std(est_taylor_losses), "mean": np.mean(est_taylor_losses), "variable": "age", "method": "taylor", "lower": np.quantile(est_taylor_losses, 0.05), "upper": np.quantile(est_taylor_losses, 0.95), "median": np.median(est_taylor_losses)})
    df_to_R.append({"delta": delta, "std": np.std(est_ipw_loss), "mean": np.mean(est_ipw_loss), "variable": "age", "method": "ipw", "lower": np.quantile(est_ipw_loss, 0.05), "upper": np.quantile(est_ipw_loss, 0.95), "median": np.median(est_ipw_loss)})
    df_to_R.append({"delta": delta, "std": 0, "mean": loss, "variable": "age", "method": "true"})

pd.DataFrame(df_to_R).to_csv(os.path.join(SAVE_PATH, 'labtest_variance_R.csv'), index=False)
