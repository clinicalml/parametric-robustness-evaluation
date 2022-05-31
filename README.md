# `ParametricRobustnessEvaluation`: Evaluate Robustness to Dataset Shifts
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


### Example
For example, consider a marginal mean shift in a Gaussian variable `W`. 
```Python
import numpy as np
from sklearn.linear_model import LinearRegression
from source.shift_gradients import ShiftLossEstimator
sle = ShiftedLossEstimator()

# Generate data and fit model
n = 100
W = np.random.normal(size=(n,1))
Y = np.sin(W) + W**2 + np.random.normal(size=(n,1))
model = LinearRegression().fit(X, Y)
# Evaluate loss per data point 
loss = (Y - model.predict(X))**2
```

We can now estimate the loss in a distribution where the mean of W shifts by 2.
```Python
estimated_loss_under_delta = sle.forward(loss_0, W, sufficient_statistic='gaussian', delta=2.0)
```
Alternatively, to estimate the loss under an arbitrary shift `delta` of magnitude smaller than `shift_strength`, 
```Python
shift_strength = 2
estimated_loss_under_shift = sle.forward(loss_0, W, sufficient_statistic='gaussian', shift_strength=shift_strength)
```
