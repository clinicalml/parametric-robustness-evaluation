import numpy as np
import pandas as pd
from tqdm import tqdm
from source.shift_gradients import ShiftLossEstimator
from sklearn.linear_model import LinearRegression
import torch


np.random.seed(1)

# Setup parameters
d_A = 3
d_X = 3
d_Y = 1
d_XY = d_X + d_Y

# Simulate coefficients
B = np.random.poisson(lam=2.0*np.ones((d_XY,)*2))
M = np.random.poisson(lam=2.0*np.ones((d_XY,d_A)))

# Simulation function
def simulate(n, delta=0.0, coef_nonlinear=0.0):
    W = np.random.normal(loc=delta, scale=1, size=(d_A, n))
    XY = np.linalg.inv(np.identity(d_XY) - B)@(M@(W + coef_nonlinear*(W**2)) + np.random.normal(loc=0, scale=1, size=(d_XY, n)))
    Z, Y = XY[:d_X, :], XY[d_X:, :]
    return torch.tensor(W.T).float(), torch.tensor(Z.T).float(), torch.tensor(Y.T).float()

# Compute IPW weights
def weights(W, delta, cap=False):
    # Fit parameters
    mu = W.mean(axis=0, keepdims=True)
    Sigma = torch.cov(W.T)
    Sigma_inv = torch.inverse(Sigma)

    # Compute weights
    num = torch.exp(-1/2*(((W - (mu + delta.T))@Sigma_inv)*(W-(mu+delta.T))).sum(axis=1, keepdim=True))
    denom = torch.exp(-1/2*(((W - mu)@Sigma_inv)*(W-mu)).sum(axis=1, keepdim=True))
    w = num/denom

    if cap: 
        w = torch.clamp(w, max=torch.quantile(w, 0.99, dim=0))
    
    return w

# Take matrix square root
def matrix_sqrt(Sigma, inverse=False):
    eigvals, U = torch.linalg.eig(torch.inverse(Sigma) if inverse else Sigma)
    return (U@torch.diag(torch.sqrt(eigvals))@U.T).float()
    

# Define shift
delta_0 = torch.ones((d_A, 1)).float()
results = []
for shift_strength in tqdm(np.linspace(0.0, 2.0, num=10)):
    delta = delta_0*shift_strength
    for coef_nonlinear in [0, 0.5]:
        A_, X_, Y_ = simulate(100000, delta=delta, coef_nonlinear=coef_nonlinear)
        for _ in range(1000):
            for n in [50, 500]:
                # Simulate data
                W, Z, Y = simulate(n, coef_nonlinear=coef_nonlinear)

                # Fit model
                model = LinearRegression()
                loss_0 = (Y - model.fit(Z=Z, y=Y).predict(Z))**2

                # Setup shift gradient
                Sigma = torch.cov(W.T)
                sle = ShiftLossEstimator()

                # Compute estimates of loss
                loss_taylor = sle.forward(loss_0, W=W,
                        sufficient_statistic=lambda W: W@matrix_sqrt(Sigma, inverse=True), delta=delta)
                loss_ipw = (loss_0 * weights(W, delta)).mean()
                loss_ipw_cap = (loss_0 * weights(W, delta, cap=True)).mean()
                loss_ = ((Y_ - model.predict(X_))**2).mean()



                results.append({"method": "IPW", "loss": loss_ipw.item(), "base": loss_.item(), "n": n, "coef_nonlinear": coef_nonlinear, "shift_strength": shift_strength})
                results.append({"method": "IPW (cap)", "loss": loss_ipw_cap.item(), "base": loss_.item(), "n": n, "coef_nonlinear": coef_nonlinear, "shift_strength": shift_strength})
                results.append({"method": "Taylor", "loss": loss_taylor.item(), "base": loss_.item(), "n": n, "coef_nonlinear": coef_nonlinear, "shift_strength": shift_strength})

df = pd.DataFrame(results)
df.to_csv("experiments/compare_ipw/compare_ipw.csv", index=False)
print(df.round(2))
