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

## Advanced use
### Conditional shift
To consider a shift in a conditional distribution `W|Z`, pass an input `Z`:
- `Z`: a numpy or torch array of shape `(n,d)` containing the parents of the shifting variable. Currently, only binary conditioning variables are supported. 

### Non-linear shift
The default shift function is `s(Z; delta) = delta`. However, for more involved shifts, one can pass functions `s_grad` and `s_hess` to the `ShiftLossEstimator` class (not the forward function),  which return the gradient and Hessian of `s` when differentiated with respect to `delta`.
- `s_grad`: Function which takes as input `Z` and outputs a `(n,d_delta,d_T)` dimensional array, for each sample point outputting the `(d_delta, d_T)` dimensional derivative of `s`, where `d_delta` is the number of parameters and `d_T` is the dimension of the sufficient statistic. For each row `(n,)`, `s_grad
- `s_hess`: Function which takes as input `Z` and outputs a `(n,d_delta,d_delta,d_T)` dimensional array, for each sample point outputting the `(d_delta, d_delta, d_T)` dimensional double derivative of `s` where `d_delta` is the number of parameters and `d_T` is the dimension of the sufficient statistic. 

### Cases when worst-case loss is larger
When finding a worst-case loss, the default setting is that a larger loss is a worst case scenario. For some loss functions, such as accuracy, a smaller value of the loss function is a worse scenario. In that case, one can set `worst_case_is_larger_loss=False`. 
- `worst_case_is_larger_loss`: bool (default = False) indicating whether adversarial shift increases loss (`True`) or decreases loss (`False`).