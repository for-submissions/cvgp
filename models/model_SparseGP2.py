
import numpy as np

import torch
from torch import nn

from torch.distributions.multivariate_normal import MultivariateNormal

from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.lazy import NonLazyTensor

from sklearn.cluster import KMeans

from utils import cholesky

class SparseGP2(nn.Module):
    def __init__(self,
            kernel='rbf', data=None,  
            posterior_data_size=None, 
            dtype=torch.float64,
            likelihood='Normal',
            inducing_point_size=100
            ):
        super(SparseGP2, self).__init__()

        self.likelihood = likelihood
        self.posterior_data_size = posterior_data_size

        self.x, self.y = data
        self.n, self.d = self.x.shape[0], self.x.shape[1]

        self.covar_module = ScaleKernel(
                RBFKernel(dtype=dtype),
                dtype=dtype
                )

        self.likelihood = likelihood
        # self.likelihood = 'Bernoulli'
        self.inducing_point_size = inducing_point_size
        x_z = self.initialize_with_kmeans()
        self.eps = 1e-4

        self.x_z = nn.Parameter(
            torch.tensor(x_z, dtype=dtype),
            requires_grad=True
            )

        self.sigma = nn.Parameter(
            # torch.randn(1, dtype=dtype),
            torch.tensor([0.26179], dtype=dtype),
            requires_grad=True
            )

    def initialize_with_kmeans(self):
        n = int(self.x.size(0) - self.inducing_point_size)
        if n <= 0:
            if abs(n) <= self.x.size(0):
                indexes = np.random.choice(
                    range(self.x.size(0)), abs(n), False
                    )
            else:
                indexes = np.random.choice(
                    range(self.x.size(0)), abs(n), True
                    )
            x_choice = self.x[indexes,:]
            x = torch.cat([self.x, x_choice], 0)
        else:
            x = self.x
        data = x
        kmeans_dict = dict()
        for _ in range(10):
            kmeans = KMeans(
                n_clusters=self.inducing_point_size,
                init='k-means++'
                ).fit(data)
            kmeans_dict[kmeans.score(data)] = kmeans
        kmeans = kmeans_dict[max(kmeans_dict)]
        return kmeans.cluster_centers_

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

    def compute_vfe_kernels(self, x):
        
        x_z = self.x_z
        device = x_z.device
        x_data, y_data = self.x.to(device), self.y.to(device)
        k_ii = self.kernel(x, x)
        k_zi = self.kernel(x_z, x) # m x i
        k_zz = self.kernel(x_z, x_z) # m x m
        k_zx = self.kernel(self.x_z, x_data)

        sigma2, _ = self.get_sigma2(k_zz.size(0))
        a = sigma2 @ k_zz + k_zx @ k_zx.T

        return k_ii, k_zi, k_zz, k_zx, a, y_data

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

    def compute_vfe_lowerbound(self, x, y):

        k_ii, k_zi, k_zz, k_zx, a, y_data = self.compute_vfe_kernels(x)

        L = cholesky(k_zz.evaluate())[0]
        alpha = torch.cholesky_solve(k_zi.evaluate(), L)
        q_xx = k_zi.T @ alpha
        sigma2, sigma2_inv = self.get_sigma2(q_xx.size(0))
        k_xx = self.kernel(x, x).evaluate()
        elbo = (
            MultivariateNormal(
                torch.zeros_like(y),
                q_xx + sigma2,
                validate_args=False
                ).log_prob(y) - 0.5 * sigma2_inv[0,0].evaluate() * torch.trace(
                    k_xx - q_xx
                    )
                    ) / self.n
        return elbo
    
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

    def approximate_predictive_distribution(self, x):

        with torch.no_grad():
            if self.likelihood == 'Normal':

                k_ii, k_zi, k_zz, k_zx, a, y_data = self.compute_vfe_kernels(x)

                L = cholesky(a.evaluate())[0]
                alpha = torch.cholesky_solve(k_zx.evaluate(), L)
                appoximate_mean = k_zi.T @ alpha @ y_data

                sigma2, _ = self.get_sigma2(k_zz.size(0))

                L1 = cholesky(k_zz.evaluate())[0]
                alpha1 = torch.cholesky_solve(k_zi.evaluate(), L1)

                alpha2 = torch.cholesky_solve(sigma2.evaluate() @ k_zi, L)

                sigma2, _ = self.get_sigma2(x.size(0))
                sigma2 = sigma2.evaluate()
                approximate_cov = k_ii - k_zi.T @ (alpha1 - alpha2)
                approximate_cov = 0.5 * (
                    approximate_cov + approximate_cov.T
                    ).evaluate() + sigma2

                posterior = MultivariateNormal(
                    appoximate_mean, approximate_cov
                    )

            return posterior

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
    #################  GAP COMPUTATIONS ####################
    ########################################################

    def posterior_inference_gap(self, x):

        dtype = self.sigma.dtype
        device = self.sigma.device
        x = torch.tensor(x, dtype=dtype).to(device)
        
        with torch.no_grad():

            exact_posterior = self.exact_predictive_distribution(x)
            approximate_posterior = self.approximate_predictive_distribution(x)

            sigma2, _ = self.get_sigma2(x.size(0))
            sigma2 = sigma2.evaluate()

            exact_mean = exact_posterior.mean
            exact_cov = exact_posterior.covariance_matrix
            chol_exact_cov = cholesky(exact_cov - sigma2)[0]
            exact_cov = chol_exact_cov @ chol_exact_cov.T
            
            approximate_mean = approximate_posterior.mean
            approximate_cov = approximate_posterior.covariance_matrix
            chol_approximate_cov = cholesky(approximate_cov - sigma2)[0]
            approximate_cov = chol_approximate_cov @ chol_approximate_cov.T
            
            p = MultivariateNormal(
                exact_mean, exact_cov
                )
            q = MultivariateNormal(
                approximate_mean, approximate_cov
                )

            n = x.size(0)

            klpq = torch.distributions.kl_divergence(p, q).item() / n
            klqp = torch.distributions.kl_divergence(q, p).item() / n

            mu_rmse = (
                exact_mean - approximate_mean
                ).pow(2).mean().pow(0.5).item()
            cov_rmse = (
                exact_cov - approximate_cov
                ).pow(2).mean().pow(0.5).item()

        return 0.5 * (klpq + klqp), klpq, klqp, mu_rmse, cov_rmse

    def hyperparameter_learning_gap(self):

        with torch.no_grad():
            device = self.sigma.device
            dtype = self.sigma.dtype
    
            x_data = torch.tensor(self.x, dtype=dtype).to(device)
            y_data = torch.tensor(self.y, dtype=dtype).to(device)

            gap =  self.compute_marginal_loglikelihood(x_data, y_data) -\
                self.compute_vfe_lowerbound(x_data, y_data)

            return gap.clamp(min=0).item()

    ########################################################
    #################### HIGH-LEVEL ########################
    ########################################################

    def predict(self, x):
        return self.approximate_predictive_distribution(x)

    def forward(self, x, y):
        loss = self.compute_vfe_lowerbound(x, y)
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