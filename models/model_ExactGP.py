
import torch
from torch import nn
# from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.lazy import NonLazyTensor

from utils import cholesky

class ExactGP(nn.Module):
    def __init__(self,
            kernel='rbf', data=None,  posterior_data_size=None, dtype=torch.float64,
            likelihood='Normal'
            ):
        super(ExactGP, self).__init__()

        self.likelihood = likelihood
        self.posterior_data_size = posterior_data_size

        self.x, self.y = data
        self.n, self.d = self.x.shape[0], self.x.shape[1]

        self.covar_module = ScaleKernel(
                RBFKernel(dtype=dtype),
                dtype=dtype
                )

        self.sigma = nn.Parameter(
            # torch.randn(1, dtype=dtype),
            torch.tensor([0.26179], dtype=dtype),
            requires_grad=True
            )

    ########################################################
    ################## KERNEL COMPUTATIONS #################
    ########################################################

    def kernel(self, x1, x2):
        cov = 0.5 * (
            self.covar_module(x1, x2) + self.covar_module(x2, x1).T
            )
        try:
            cov = cov + self.eps * torch.eye(cov.size(-1)).to(cov.device)
        except:
            cov = cov
        return cov
    
    def compute_exact_kernels(self, x_star, x_data, y_data):

        sigma2, sigma2_inv = self.get_sigma2(y_data.size(0))

        k_ii = self.kernel(x_star, x_star)
        k_xi = self.kernel(x_data, x_star) # n x i
        k_xx = self.kernel(x_data, x_data) # m x m

        a = sigma2 + k_xx

        return k_ii, k_xi, k_xx, a

    ########################################################
    ################## LOSS COMPUTATIONS ###################
    ########################################################

    def compute_marginal_loglikelihood(self, x, y):
        sigma2, sigma2_inv = self.get_sigma2(y.size(0))

        k_xx = self.kernel(x, x)

        loglikelihood = MultivariateNormal(
                torch.zeros_like(y),
                (k_xx + sigma2).evaluate()
                ).log_prob(y) / y.size(0)

        return loglikelihood

    
    ########################################################
    ######### PREDICTIVE DISTRIBUTION COMPUTATIONS #########
    ########################################################
    
    def exact_predictive_distribution(self, x):
        with torch.no_grad():
            if self.likelihood == 'Normal':
                device = self.sigma.device
                dtype = self.sigma.dtype

                x_data = torch.tensor(self.x, dtype=dtype).to(device)
                y_data = torch.tensor(self.y, dtype=dtype).to(device)

                k_ii, k_xi, k_xx, a = self.compute_exact_kernels(
                         x, x_data, y_data
                         )

                sigma2, _ = self.get_sigma2(x.size(0))
                L = cholesky(a.evaluate())[0]
                alpha = torch.cholesky_solve(k_xi.evaluate(), L)
                exact_cov = k_ii - k_xi.T @ alpha
                exact_cov = 0.5 * (exact_cov + exact_cov.T) + sigma2
                exact_cov = exact_cov.evaluate()

                alpha = torch.cholesky_solve(y_data.unsqueeze(1), L)
                exact_mean = torch.squeeze(k_xi.T @ alpha, -1)

                posterior = MultivariateNormal(exact_mean, exact_cov)

        return posterior

    ########################################################
    #################### HIGH-LEVEL ########################
    ########################################################

    def predict(self, x):
        return self.exact_predictive_distribution(x)

    def forward(self, x, y):
        loss = self.compute_marginal_loglikelihood(x, y)
        with torch.no_grad():
            predictive_distribution = self.predict(x)
        return loss, predictive_distribution

    ########################################################
    ####################### UTILS ##########################
    ########################################################

    def get_sigma2(self, size):
        sigma = nn.Softplus()(self.sigma)
        sigma2 = sigma.pow(2).repeat_interleave(size, 0)
        sigma2_inv = sigma2.pow(-1)
        return NonLazyTensor(torch.diag(sigma2)),\
            NonLazyTensor(torch.diag(sigma2_inv))
