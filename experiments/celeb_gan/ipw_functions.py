from scipy.optimize import minimize
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ipw_weights_single(delta, data, child, cpd, order=0):
    parents = cpd[child]['Parents']
    parents_sorted = sorted(parents.keys())
    base_coef = cpd[child]['Base']
    d = len(parents)
    n = data.shape[0]
    
    if parents:
        groups = np.array(data[parents_sorted] @ np.array([2**j for j in range(d)])).astype(int)
    else:
        groups = np.zeros(n).astype(int)

    # Get obs probability
    eta_obs = base_coef + np.sum([data[parent].values*parents[parent] for parent in parents_sorted], axis=0)
    ratio = np.exp(data[child]*delta[groups]) * (1 + np.exp(eta_obs))/(1 + np.exp(eta_obs + delta[groups]))
    if order == 0:
        return ratio
    elif order == 1:
        resid = data[child] - sigmoid(eta_obs + delta[groups])
        deriv_s = np.zeros((n, 2**d))
        deriv_s[np.arange(n), groups] = 1
        return np.einsum('ij,i->ij', deriv_s, resid)
    elif order == 2:
        resid = data[child] - sigmoid(eta_obs + delta[groups])
        sq_resid = resid**2 - sigmoid(eta_obs + delta[groups])*(1 - sigmoid(eta_obs + delta[groups]))
        deriv_s = np.zeros((n, 2**d))
        deriv_s[np.arange(n), groups] = 1
        return np.einsum('ij, i, ik -> ijk', deriv_s, sq_resid, deriv_s)


def ipw_weights_all(delta, data, cpd, order=0):
    # Split delta up corresponding to each probability distribution
    children_sorted = sorted(data.drop("Male", axis=1).columns)
    splits = np.cumsum([2**len(cpd[child]['Parents']) for child in children_sorted])
    deltas = np.split(delta, splits)[:-1]

    child_weights = {}
    for child, delta in zip(children_sorted, deltas):
        w =  ipw_weights_single(delta, data, child, cpd, order=0)
        child_weights[child] = w
    
    delta_childs = list(zip(deltas, children_sorted))
    weight_0 = np.prod([w for w in child_weights.values()], axis=0)
    weights_1 = {child: ipw_weights_single(delta, data, child, cpd, order=1) for delta, child in delta_childs}

    if order == 0:
        return weight_0
    elif order == 1:
        return np.einsum('i, ik->ik', weight_0, np.concatenate([ipw_weights_single(delta, data, child, cpd, order=1) for delta, child in delta_childs], axis=1))
    elif order == 2:
        weights_2 = {child: ipw_weights_single(delta, data, child, cpd, order=2) for delta, child in delta_childs}
        v = np.concatenate([np.concatenate([np.einsum('ij, ik->ijk', weights_1[child_1], weights_1[child_2]) if child_1 != child_2 else weights_2[child_1]
                                            for child_2 in data.drop('Male', axis=1)], axis=2) for child_1 in data.drop('Male', axis=1)], axis=1)

        return np.einsum('i, ijk->ijk', weight_0, v)

def ipw(delta, data, acc, cpd):
    w = ipw_weights_all(delta, data, cpd, order=0)
    return (w*acc).mean()

def ipw_grad(delta, data, acc, cpd):
    w = ipw_weights_all(delta, data, cpd, order=1)
    return (np.einsum('ij, i->ij', w, acc)).mean(axis=0)

def ipw_hess(delta, data, acc, cpd):
    w = ipw_weights_all(delta, data, cpd, order=2)
    return (np.einsum('ijk, i->ijk', w, acc)).mean(axis=0)

def optimize_ipw(data, acc, cpd, args):
    cons = ({'type': 'ineq', 'fun': lambda x:  args.shift_strength**2-sum(x**2) })
    delta_init = np.zeros(31)

    if args.method == "SLSQP":
        optim = minimize(ipw, delta_init, args=(data, acc, cpd), method='SLSQP', constraints=cons, jac=ipw_grad, options={'disp': True})
    else:
        optim = minimize(ipw, delta_init, args=(data, acc, cpd), method='trust-constr', constraints=cons, jac=ipw_grad, hess=ipw_hess, options={'disp': True})
    delta = optim['x']
    return delta
