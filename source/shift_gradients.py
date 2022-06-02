from pandas import infer_freq
import torch
import numpy as np
import trustregion

def cov(m1, m2, Z=None):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `Z = [x_1, x_2, ..., x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m1, m2: W 1-D or 2-D array containing multiple variables and observations.
            Each column of `m` represents a variable, and each row a single
            observation of all those variables.
    Returns:
        The covariance matrix of the variables.
    '''
    if m1.dim() > 1:
        raise ValueError('m1 has more than 1 dimensions')
    if m2 is None:
        m2 = m1
    # Note we do allow for m2 to have more dimensions. 
    # We don't allow this for m1, because einsum can't handle it.
    if m2.dim() < 2:
        m2 = m2.view(-1, 1)
    if m2.shape[0] != m1.shape[0]:
        raise ValueError('m1 and m2 must have the same number of rows')

    if Z is None:
        mean_1 = m1.mean(axis=0, keepdims=True)
        mean_2 = m2.mean(axis=0, keepdims=True)
    else:
        mean_1 = cond_mean_given_binary(m1, Z)
        mean_2 = cond_mean_given_binary(m2, Z)
        
    fact = 1.0 / (m1.size(0) - 1)
    m1 = m1 - mean_1
    m2 = m2 - mean_2

    tmp = torch.einsum("i,ij...->ij...", m1, m2)
    return fact*tmp


def cond_mean_given_binary(W, Z):
    """
    Compute the conditional mean of W given Z, where Z is a binary matrix.
    Output is the vector E[W|Z], meaning that each entry can only take one of 2^d_Z values, where d_Z is the number of columns in Z.
    """
    if Z.dim() > 2:
        raise ValueError('Conditioning variable Z has more than 2 dimensions')
    elif Z.dim() < 2:
        Z = Z.unsqueeze(-1)
    
    # Compute the conditional mean vector of W given Z, where Z is a binary matrix.
    groups = (Z @ torch.diag(torch.tensor([2.0**j for j in range(Z.shape[1])])).sum(axis=1)).long()
    weights = torch.zeros((2**Z.shape[1], len(W)))
    weights[groups, torch.arange(len(W))] = 1
    weights = torch.nn.functional.normalize(weights, p=1, dim=1)
    group_means = torch.einsum('ij,j...->i...', weights, W)

    return group_means[groups]

def convert_np_to_torch(x):
    """
    Convert a numpy array to a torch tensor.
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    # Check if x is a float or int (but neither numpy nor torch)
    elif isinstance(x, (int, float)):
        return torch.tensor(x).float()
    else:
        return x.float()

def gaussian_sufficient_statistic(W):
    """
    Implement the sufficient statistic of a Gaussian random variable with known mean
    """
    if W.dim() == 1:
        sigma = torch.sqrt(torch.var(W))
    else:
        sigma = torch.sqrt(torch.var(W, dim=0, keepdim=True))
    return W/sigma

class ShiftLossEstimator(torch.nn.Module):
    """
    Compute robust loss function.
    """
    def __init__(self, s_grad=None, s_hess=None):
        super().__init__()
        self.s_grad = s_grad
        self.s_hess = s_hess
        self.sg1 = None
        self.sg2 = None
        self.delta = None
    
    def forward(self, loss_0, W, sufficient_statistic, Z=None, delta=None, shift_strength=1.0, use_stored_sg=False, worst_case_is_larger_loss=True):
        # If only single elements passed, convert to list
        if not isinstance(W, list):
            W = [W]
        if not isinstance(sufficient_statistic, list):
            # If only a single function/string is passed, convert to list of same length as W
            sufficient_statistic = [sufficient_statistic] * len(W)
        if not isinstance(Z, list):
            # If only a single None is passed, convert to list of None's of same length as W
            if Z is None:
                Z = [None] * len(W)
            else: 
                Z = [Z]
        if not delta is None:
            if not isinstance(delta, list):
                delta = [delta]
            delta = [convert_np_to_torch(d) for d in delta]
            if len(delta) > 1:
                delta = torch.cat(delta, axis=0)
            else:
                delta = torch.tensor(delta)
            delta = delta.view(-1, 1).float()
        assert len(sufficient_statistic) == len(W) == len(Z)


        # Convert to torch
        loss_0 = convert_np_to_torch(loss_0)
        W = [convert_np_to_torch(w) for w in W]
        Z = [convert_np_to_torch(z) for z in Z]
        
        # Check that each element in Z is a binary matrix
        for z in Z:
            if z is not None:
                # if len(torch.unique(z)) > 2:
                close_1 = torch.isclose(z, torch.ones(z.shape))
                close_0 = torch.isclose(z, torch.zeros(z.shape))
                if not all(torch.logical_or(close_1, close_0)):
                    raise ValueError('Z must be a binary matrix containing only 0 and 1')
                

        # If strings are passed for the sufficient_statistic, convert to functions
        for i, ss in enumerate(sufficient_statistic):
            if isinstance(ss, str):
                if ss in ['binary', 'binomial', 'poisson', 'exponential']:
                    sufficient_statistic[i] = lambda x: x
                elif ss in ['gaussian', 'normal']:
                    sufficient_statistic[i] = gaussian_sufficient_statistic
                else: 
                    raise ValueError('Unknown sufficient statistic')

        # Compute shift gradient at current loss of predictor
        if self.sg1 is not None and use_stored_sg:
            sg1 = self.sg1
        else:
            sg1_list = [self.get_sg1(loss_0.squeeze(), W=w, Z=z, sufficient_statistic=ss) for w, ss, z in zip(W, sufficient_statistic, Z)]
            sg1 = torch.cat(sg1_list, axis=0)
            self.sg1 = sg1

        if self.sg2 is not None and use_stored_sg:
            sg2 = self.sg2
        else:
            sg2_list = [self.get_sg2(loss_0.squeeze(), W=w, Z=z, sufficient_statistic=ss) for w, ss, z in zip(W, sufficient_statistic, Z)]
            sg2 = torch.block_diag(*sg2_list)
            self.sg2 = sg2
        

        # Find worst case direction delta
        if delta is None:
            scale = -1.0 if worst_case_is_larger_loss else 1.0

            delta = trustregion.solve(scale*sg1.detach().numpy(), scale*sg2.detach().numpy(), shift_strength)
            delta = torch.from_numpy(delta).float().view(-1, 1)
        
        # Store worst-case direction
        self.delta = delta

        # Compute loss approximations at worst case direction
        loss_1 = (sg1.view(-1, 1).T@delta).squeeze()
        loss_2 = (delta.T@sg2@delta).squeeze()
        
        return torch.mean(loss_0) + loss_1 + loss_2/2.0

    def get_sg1(self, loss_0, W, sufficient_statistic, Z=None):
        ss = sufficient_statistic(W=W)
        cond_cov = cov(loss_0, ss, Z=Z) 
        if self.s_grad is None:
            # collapses (summing over) the first dimension
            out = torch.einsum("i...->...", cond_cov)
        else:
            s_grad_ = self.s_grad(Z=Z)
            # First dimension ("i") is data, then matrix/vector  multiplication
            # as in Theorem 1.
            out = torch.einsum("ijk, ik...->j...", s_grad_, cond_cov)
        return out

    def get_sg2(self, loss_0, W, sufficient_statistic, Z=None):
        ss = sufficient_statistic(W=W)
        if Z is None:
            mean_ss = ss.mean(axis=0, keepdims=True)
        else:
            mean_ss = cond_mean_given_binary(ss, Z)
        ss = ss - mean_ss
        # Outer product
        ss = torch.einsum("ij,ik->ijk", ss, ss)
        cond_cov = cov(loss_0, ss, Z=Z)
        if self.s_hess is None:
            out = torch.einsum("i...->...", cond_cov)
        else:
            s_hess_ = self.s_hess(Z=Z)
            out = torch.einsum("ikst,itj,isvj->kv", s_hess_, cond_cov, s_hess_.transpose(1, 2))
        return out