from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
import ssm.stats as stats
from autograd import grad, hessian
from autograd.misc import flatten
from scipy.optimize import minimize
# Import generic useful functions from ssm package
from ssm.util import ensure_args_are_lists


# lapse model
class lapse_model(object):
    def __init__(self, M, num_lapse_params, include_bias=True):
        self.M = M
        self.include_bias = include_bias
        # Parameters linking input to state distribution
        if self.include_bias:
            self.W = 2 * npr.randn(M + 1)
        else:
            self.W = 2 * npr.randn(M)
        self.num_lapse_params = num_lapse_params
        # Lapse rates - initialize gamma and lambda lapse rates as 0.05 with
        # some noise
        # Ensure that gamma and lambda parameters are greater than or equal
        # to 0 and less than or equal to 1
        gamma = np.maximum(0.05 + 0.03 * npr.rand(1)[0], 0)
        gamma = np.minimum(gamma, 1)
        if num_lapse_params == 2:
            lamb = np.maximum(0.05 + 0.03 * npr.rand(1)[0], 0)
            lamb = np.minimum(lamb, 1)
            self.lapse_params = np.array([gamma, lamb])
        else:
            self.lapse_params = np.array([gamma])

    @property
    def params(self):
        return [self.W, self.lapse_params]

    @params.setter
    def params(self, value):
        self.W = value[0]
        self.lapse_params = value[1]

    def log_prior(self):
        return 0

    def calculate_pr_lapse(self):
        if self.num_lapse_params == 2:
            pr_lapse = self.lapse_params[0] + self.lapse_params[1]
        else:
            pr_lapse = 2 * self.lapse_params[0]
        return pr_lapse

    # Calculate probability right at each time step
    # This is pr(right) = gamma + (1-lambda-gamma)(1/(1+e^(-wx)))
    def calculate_pr_right(self, input):
        # Update input to include offset term:
        if self.include_bias:
            input = np.append(input, np.ones((input.shape[0], 1)), axis=1)
        logits = np.dot(self.W, input.T)
        softmax = np.exp(logits) / (1 + np.exp(logits))
        if self.num_lapse_params == 2:
            prob_right = self.lapse_params[0] + (
                    1 - self.lapse_params[0] - self.lapse_params[1]) * softmax
        else:
            prob_right = self.lapse_params[0] + (
                    1 - 2 * self.lapse_params[0]) * softmax
        return prob_right, softmax

    # Calculate time dependent logits - output is matrix of size Tx2,
    # with pr(R) in 2nd column
    # Input is size TxM
    def calculate_logits(self, input):
        prob_right, _ = self.calculate_pr_right(input)
        assert (max(prob_right) <= 1) or (
                min(
                    prob_right) >= 0), 'At least one of the probabilities is '\
                                       'not between 0 and 1'
        # Now calculate prob_left
        prob_left = 1 - prob_right
        # Calculate logits - array of size Tx2 with log(prob_left) as first
        # column and log(prob_right) as second column
        time_dependent_logits = np.transpose(
            np.vstack((np.log(prob_left), np.log(prob_right))))
        # Add in lapse parameters
        return time_dependent_logits

        # Calculate log-likelihood of observed data: LL = sum_{i=0}^{T}(y_{
        # i}log(p_{i}) + (1-y_{i})log(1-p_{i})) where y_{i} is a one-hot
        # vector of the data

    def log_likelihoods(self, data, input, mask, tag):
        time_dependent_logits = self.calculate_logits(input)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.categorical_logpdf(data[:, None, :],
                                        time_dependent_logits[:, None,
                                        None, :],
                                        mask=mask[:, None, :])

    def sample(self, input, tag=None, with_noise=True):
        # Sample both a state and a choice at each time point:
        T = input.shape[0]
        data_sample = np.zeros(T, dtype='int')
        z_sample = np.zeros(T, dtype='int')
        pr_right, softmax = self.calculate_pr_right(
            input)  # vectors of length T
        pr_lapse = self.calculate_pr_lapse()
        for t in range(T):
            z_sample[t] = npr.choice(2, p=[(1 - pr_lapse), pr_lapse])
            if z_sample[t] == 0:
                data_sample[t] = npr.choice(2,
                                            p=[(1 - softmax[t]), softmax[t]])
            else:  # indicates a lapse
                if self.num_lapse_params == 1:
                    data_sample[t] = npr.choice(2, p=[0.5, 0.5])
                else:
                    lapse_pr_right = self.lapse_params[0] / (
                            self.lapse_params[0] + self.lapse_params[1])
                    data_sample[t] = npr.choice(2,
                                                p=[(1 - lapse_pr_right),
                                                   lapse_pr_right])
        data_sample = np.expand_dims(data_sample, axis=1)
        return z_sample, data_sample

    # log marginal likelihood of data
    @ensure_args_are_lists
    def log_marginal(self, datas, inputs, masks, tags):
        elbo = self.log_prior()
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            lls = self.log_likelihoods(data, input, mask, tag)
            elbo += np.sum(lls)
        return elbo

    @ensure_args_are_lists
    def fit_lapse_model(self,
                        datas,
                        inputs,
                        masks,
                        tags,
                        optimizer="s",
                        num_iters=1000,
                        **kwargs):

        # define optimization target
        def _objective(params, itr):
            self.params = params
            obj = self.log_marginal(datas, inputs, masks, tags)
            return -obj

        # Update params as output of optimizer
        self.params, self.hessian = minimize_loss(_objective,
                                                  self.params,
                                                  self.num_lapse_params,
                                                  num_iters=num_iters,
                                                  **kwargs)


def minimize_loss(loss,
                  x0,
                  num_lapse_params,
                  verbose=False,
                  num_iters=1000):
    # Flatten the loss
    _x0, unflatten = flatten(x0)
    _objective = lambda x_flat, itr: loss(unflatten(x_flat), itr)

    # Specify callback for fitting
    itr = [0]

    def callback(x_flat):
        itr[0] += 1
        print("Iteration {} loss: {:.3f}".format(itr[0],
                                                 loss(unflatten(x_flat), -1)))
        print("Grad: ")
        grad_to_print = grad(_objective)(x_flat, -1)
        print(grad_to_print)

    # Bounds
    N = x0[0].shape[0]
    if num_lapse_params == 2:
        bounds = [(-10, 10) for i in range(N + 2)]
        bounds[N] = (0, 0.5)
        bounds[N + 1] = (0, 0.5)
    else:
        bounds = [(-10, 10) for i in range(N + 1)]
        bounds[N] = (0, 0.5)
    # Call the optimizer
    result = minimize(_objective,
                      _x0,
                      args=(-1,),
                      jac=grad(_objective),
                      method="SLSQP",
                      bounds=bounds,
                      callback=callback if verbose else None,
                      options=dict(maxiter=num_iters, disp=verbose))
    if verbose:
        print("Optimization completed with message: \n{}".format(
            result.message))

    if not result.success:
        warn("Optimization failed with message:\n{}".format(result.message))

    # Get hessian:
    autograd_hessian = hessian(_objective)
    hess_to_return = autograd_hessian(result.x, -1)
    return unflatten(result.x), hess_to_return
