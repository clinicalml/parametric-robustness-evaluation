# `EvaluateRobustness`: Evaluate Robustness to Dataset Shifts
This package implements the second-order approximation described in [Evaluating Robustness to Dataset Shift via Parametric Robustness Sets](). 


## Use

### Inputs 
To use on a data set with `n` samples, we assume the following input
- `loss`: a numpy or torch array of shape `(n,)` containing prediction loss of each individual dataset. 
    - e.g. to evaluate accuracy under a shift, define `loss = 1.0*(Y == model(X))` and to evaluate the MSE define `loss = (Y - model(X))**2`.
- `W`: a numpy or torch array of shape `(n,d)` containing the variable(s) that shift. 
- `sufficient_statistic`: Either of
    1. a string in `['gaussian', 'binary', 'binomial', 'poisson', 'exponential']`, which loads the relevant sufficient statistic. 
    2. a function that takes as input `W` and outputs the sufficient statistic of `W`. See [here](https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions) for a table of sufficient statistics. For example, if the sufficient statistic is `T(W) = W`, define `sufficient_statistic = lambda W: W`. 


```Python
import numpy as np
from source.shift_gradients import ShiftLossEstimator
from sklearn.linear_model import LinearRegression
sle = ShiftedLossEstimator()

# Generate data and fit model
n = 100
W = np.random.normal(size=(n,1))
Y = np.sin(W) + W**2 + np.random.normal(size=(n,1))
model = LinearRegression().fit(X, Y)

# Evaluate loss per data point 
loss = (Y - model.predict(X))**2

# Estimate loss for a fixed shift delta, e.g. a shift in mean of X by 2
#TODO: Add sufficient_statistic
estimated_loss_under_delta = sle.forward(loss_0, W, delta=2.0)
```
Alternatively, to estimate the loss under an arbitrary shift `delta` of magnitude smaller than `shift_strength`, 
```Python
estimated_loss_under_shift = sle.forward(loss_0, W, shift_strength=shift_strength)
```
