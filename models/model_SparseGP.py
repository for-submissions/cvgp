
import torch
import gpytorch

from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel

from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np
from sklearn.cluster import KMeans

from utils import cholesky


class SparseGP(gpytorch.models.ExactGP):
    def __init__(self,
            kernel='rbf',
            loss='SVGP', data=None,
            dtype=torch.float64,
            inducing_point_size=100
        ):

        # No need to keep data as attributes
        self.x, self.y = data
        self.n, self.d = self.x.shape[0], self.x.shape[1]

        # Gaussian Likelihood
        likelihood=gpytorch.likelihoods.GaussianLikelihood()

        # Init
        super(SparseGP, self).__init__(self.x, self.y, likelihood)

        # GP mean
        self.mean_module = ZeroMean()

        # GP kernel
        if kernel == 'rbf':
            self.covar_module = ScaleKernel(RBFKernel())
        else:
            raise ValueError('Unknown kernel function')

        # Initialize inducing points
        self.inducing_point_size=inducing_point_size

        # Sparse GP Model is based on simply wrapping base covar in an InducingPointKernel
        self.covar_module_ = InducingPointKernel(
            self.covar_module,
            inducing_points=torch.tensor(
                self.initialize_with_kmeans(),
                dtype=dtype
                ),
            likelihood=likelihood
        )

        # Training loss information, for later use
        # Loss function is the marginal log likelihood in Titsias by gpytorch
        self.loss_f=gpytorch.mlls.ExactMarginalLogLikelihood

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

    def compute_exact_kernels(self, x_star, x_data, y_data):

        sigma2 = self.likelihood.noise
        sigma2 = torch.eye(x_data.size(0)).to(sigma2.device) * sigma2

        k_ii = self.covar_module(x_star, x_star)
        k_xi = self.covar_module(x_data, x_star) # n x i
        k_xx = self.covar_module(x_data, x_data) # m x m

        a = sigma2 + k_xx

        return k_ii, k_xi, k_xx, a

    ########################################################
    ################## LOSS COMPUTATIONS ###################
    ########################################################

    def compute_marginal_loglikelihood(self, x, y):
        
        sigma2 = self.likelihood.noise
        sigma2 = torch.eye(x.size(0)).to(sigma2.device) * sigma2
        
        k_xx = self.covar_module(x, x)
        
        loglikelihood = MultivariateNormal(
                torch.zeros_like(y),
                (k_xx + sigma2).evaluate()
                ).log_prob(y) / y.size(0)
        
        return loglikelihood

    def compute_svgp_lowerbound(self, x, y):
        elbo = self.loss_f(
            self.likelihood,
            self,
            num_data=x.shape[0]
            )(self.forward(x), y)
        return elbo

    ########################################################
    ######### PREDICTIVE DISTRIBUTION COMPUTATIONS #########
    ########################################################

    def approximate_predictive_distribution(self, x):
        with torch.no_grad():
                posterior = self(x)
                #does not include noise in y - gpytorch
        return posterior

    def exact_predictive_distribution(self, x):
        with torch.no_grad():

            device = self.likelihood.noise.device
            dtype = self.likelihood.noise.dtype

            x_data = torch.tensor(self.x, dtype=dtype).to(device)
            y_data = torch.tensor(self.y, dtype=dtype).to(device)

            k_ii, k_xi, k_xx, a = self.compute_exact_kernels(
                     x, x_data, y_data
                     )

            L = cholesky(a.evaluate())[0]
            alpha = torch.cholesky_solve(k_xi.evaluate(), L)
            exact_cov = k_ii - k_xi.T @ alpha
            exact_cov = 0.5 * (exact_cov + exact_cov.T)
            exact_cov = exact_cov.evaluate()

            alpha = torch.cholesky_solve(y_data.unsqueeze(1), L)
            exact_mean = torch.squeeze(k_xi.T @ alpha, -1)

            posterior = MultivariateNormal(exact_mean, exact_cov)

        return posterior

    ########################################################
    #################  GAP COMPUTATIONS ####################
    ########################################################

    def posterior_inference_gap(self, x):
        
        dtype = self.likelihood.noise.dtype
        device = self.likelihood.noise.device
        x = torch.tensor(x, dtype=dtype).to(device)
        
        with torch.no_grad():

            exact_posterior = self.exact_predictive_distribution(x)
            approximate_posterior = self.approximate_predictive_distribution(x)

            exact_mean = exact_posterior.mean
            exact_cov = exact_posterior.covariance_matrix
            chol_exact_cov = cholesky(exact_cov)[0]
            exact_cov = chol_exact_cov @ chol_exact_cov.T
            
            approximate_mean = approximate_posterior.mean
            approximate_cov = approximate_posterior.covariance_matrix
            chol_approximate_cov = cholesky(approximate_cov)[0]
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
            device = self.likelihood.noise.device
            dtype = self.likelihood.noise.dtype

            x_data = torch.tensor(self.x, dtype=dtype).to(device)
            y_data = torch.tensor(self.y, dtype=dtype).to(device)

            gap =  self.compute_marginal_loglikelihood(x_data, y_data) -\
                self.compute_svgp_lowerbound(x_data, y_data)

            return gap.clamp(min=0).item()

    ########################################################
    #################### HIGH-LEVEL ########################
    ########################################################

    def forward(self, x):

        # Mean and covariance forward
        mean_x = self.mean_module(x)
        covar_x = self.covar_module_(x)
        
        # if not self.training:
        
        #     device = self.likelihood.noise.device
        #     dtype = self.likelihood.noise.dtype
            
        #     x_data = torch.tensor(self.x, dtype=dtype).to(device)
        #     y_data = torch.tensor(self.y, dtype=dtype).to(device)
            
        #     k_ii = self.covar_module(
        #         x,
        #         x
        #         ).evaluate()
        #     k_zi = self.covar_module(
        #         self.covar_module.inducing_points,
        #         x
        #         ).evaluate()
    
        #     k_zz = self.covar_module(
        #         self.covar_module.inducing_points,
        #         self.covar_module.inducing_points
        #         ).evaluate()
    
        #     k_zx = self.covar_module(
        #         self.covar_module.inducing_points,
        #         x_data
        #         ).evaluate()
            
        #     sigma2 = self.likelihood.noise
        #     sigma2 = torch.eye(k_zz.size(0)).to(sigma2.device) * sigma2
        #     sigma2_inv = sigma2.inverse()
            
        #     mean_x = k_zi.T @ torch.cholesky_inverse(
        #         cholesky(sigma2 @ k_zz + k_zx @ k_zx.T)[0]
        #         ) @ k_zx @ y_data       
            
        #     k_zz_inv = torch.cholesky_inverse(
        #         cholesky(k_zz)[0]
        #         )
            
        #     covar_x = k_ii - k_zi.T @ (
        #         k_zz_inv - torch.cholesky_inverse(
        #             cholesky(k_zz + sigma2_inv @ k_zx @ k_zx.T)[0]
        #             )
        #         ) @ k_zi 
         
        # Output is Multivariate Gaussian
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)