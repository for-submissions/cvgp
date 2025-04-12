
import numpy as np

import torch
from torch import nn

from gpytorch.distributions import MultivariateNormal

from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.lazy import NonLazyTensor # just a wrapper
# import scipy
from sklearn.cluster import KMeans

import xgboost as xgb

from utils import cholesky


class CVTGP(nn.Module):
    def __init__(self,
                 kernel='rbf', data=None,
                 inducing_point_size=100, dtype=torch.float64,
                 ):
        super(CVTGP, self).__init__()


        self.inducing_point_size = inducing_point_size

        # need to cap beta and sigma and beta - we can't let them be 0
        # exactly for numerical reasons
        # self.eps = 1e-4

        self.x, self.y = data
        self.n, self.d = self.x.shape[0], self.x.shape[1]

        x_c, y_c, beta = self.initialize_with_kmeans()
        self.beta = nn.Parameter(
            torch.tensor(beta, dtype=dtype),
            requires_grad=True
        )
        self.x_c = nn.Parameter(
            torch.tensor(x_c, dtype=dtype),
            requires_grad=True
        )

        self.y_c = nn.Parameter(
            torch.tensor(y_c, dtype=dtype),
            requires_grad=True
        )

        #0.26179 softplus corresponds to Gpytorch likelihood noise init
        self.sigma = nn.Parameter(
            torch.tensor([0.26179], dtype=dtype),
            requires_grad=True
        )
        self.covar_module = ScaleKernel(
            RBFKernel(dtype=dtype),
            dtype=dtype
        )

    def initialize_with_kmeans(self):
        #For cvgp, we have to initialize coreset weights \beta nad coreset
        #and pseudo-outputs along with pseudo-inputs (inducing points)
        #we use kmeans centroids for beta and "a machine learning model" to
        #find y_c (or Y_M w.r.t. paper) that corresponds to kmeans centroids
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
            x_choice = self.x[indexes, :]
            x = torch.cat([self.x, x_choice], 0)
        else:
            x = self.x

        data = x.numpy()
        kmeans_dict = dict()
        for _ in range(10):
            kmeans = KMeans(
                n_clusters=self.inducing_point_size,
                init='k-means++'
            ).fit(data)
            kmeans_dict[kmeans.score(data)] = kmeans
        kmeans = kmeans_dict[max(kmeans_dict)]
        data = kmeans.cluster_centers_
        beta_ = kmeans.labels_
        key, value = np.unique(beta_, return_counts=True)
        value = value / value.sum()
        beta = kmeans.predict(data)
        beta_dict = {k: v for (k, v) in zip(key, value)}
        beta = np.asarray([beta_dict[b] for b in beta]) * self.x.size(0)

        interpolator = xgb.XGBRegressor().fit(
            self.x.numpy(),
            self.y.numpy(),
        )
        x_, y_ = data, interpolator.predict(data)

        return x_, y_, beta

    ########################################################
    ################## KERNEL COMPUTATIONS #################
    ########################################################

    def kernel(self, x1, x2):
        # cov = 0.5 * (
        #         self.covar_module(x1, x2) + self.covar_module(x2, x1).T
        # )
        # try:
        #     cov = cov + self.eps * torch.eye(cov.size(-1)).to(cov.device)
        # except:
        #     cov = cov
        return self.covar_module(x1, x2)

    def compute_exact_kernels(self, x_star, x_data, y_data):

        sigma2, sigma2_inv = self.get_sigma2(y_data.size(0))

        k_ii = self.kernel(x_star, x_star)
        k_xi = self.kernel(x_data, x_star)  # n x i
        k_xx = self.kernel(x_data, x_data)  # m x m

        a = sigma2 + k_xx

        return k_ii, k_xi, k_xx, a

    def compute_cvtgp_kernels(self, x):

        x_c, y_c = self.x_c, self.y_c
        beta, beta_inv = self.get_beta(y_c.size(0))
        sigma2, sigma2_inv = self.get_sigma2(y_c.size(0))

        k_ii = self.kernel(x, x)
        k_ci = self.kernel(x_c, x)  # m x i
        k_cc = self.kernel(x_c, x_c)  # m x m
        sigmabetac = beta_inv @ sigma2

        a = sigmabetac + k_cc

        return k_ii, k_ci, k_cc, beta, a, y_c

    ########################################################
    ################## LOSS COMPUTATIONS ###################
    ########################################################

    def compute_expected_value(self, x, y):
        """
        E_q(f)[\log p(y|f)].
        Computes the predictive part of the loss function that will be
        combined with the kl term.
        """
        sigma2, sigma2_inv = self.get_sigma2(y.size(0))
        k_ii, k_ci, k_cc, beta, a, y_c = self.compute_cvtgp_kernels(
            x
        )

        """
        If normal likelihood, we may compute this expectation analytically.
        """
        k_ci_eval = k_ci.evaluate()
        term_1 = torch.squeeze(
            k_ci.T @ a.inv_matmul(y_c)
        )
        q_fnc = torch.log(
            2 * np.pi * torch.diagonal(sigma2)
        ) + torch.diagonal(sigma2_inv) * (
                        y.pow(2) - 2 * y * term_1 + term_1.pow(2)
                )
        q_fnc += torch.diagonal(sigma2_inv) * (
                torch.diagonal(
                    k_ii
                ) - torch.sum(
            k_ci * a.inv_matmul(k_ci_eval), 0
        )
        )

        q_fnc = - 0.5 * q_fnc
        q_fnc = torch.mean(q_fnc, 0)

        return q_fnc, (k_cc, beta, y_c, a)

    def compute_cvtgp_lowerbound(self, x, y):
        q_fnc, (k_cc, beta, y_c, a) = self.compute_expected_value(x, y)
        sigma2, sigma2_inv = self.get_sigma2(beta.size(0))
        mu2 = y_c @ a.inv_matmul(k_cc.evaluate()) @ a.inv_matmul(y_c)  # done
        tr = torch.trace(a.inv_matmul(k_cc.evaluate()))  # done
        log_det = torch.logdet(a) - torch.logdet(sigma2) + torch.logdet(beta)
        elbo = q_fnc - 0.5 * (-tr + mu2 + log_det) / self.n
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
            sigma2, sigma2_inv = self.get_sigma2(x.size(0))

            k_ii, k_ci, k_cc, beta, a, y_c = self.compute_cvtgp_kernels(x)

            L = cholesky(a.evaluate())[0]
            alpha = torch.cholesky_solve(y_c.unsqueeze(-1), L)
            appoximate_mean = (k_ci.T @ alpha).squeeze(-1)
            # appoximate_mean = k_ci.T @ a.inv_matmul(y_c)
            alpha = torch.cholesky_solve(k_ci.evaluate(), L)
            approximate_cov = k_ii - k_ci.T @ alpha
            approximate_cov = 0.5 * (
                    approximate_cov + approximate_cov.T
            )
            approximate_cov = approximate_cov + sigma2

            posterior = MultivariateNormal(
                appoximate_mean, approximate_cov
            )

            return posterior

    def exact_predictive_distribution(self, x):
        with torch.no_grad():
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

            gap = self.compute_marginal_loglikelihood(x_data, y_data) - \
                  self.compute_cvtgp_lowerbound(x_data, y_data)

            return gap.clamp(min=0).item()

    ########################################################
    #################### HIGH-LEVEL ########################
    ########################################################

    def predict(self, x):
        return self.approximate_predictive_distribution(x)

    def forward(self, x, y):
        loss = self.compute_cvtgp_lowerbound(x, y)
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
        return NonLazyTensor(torch.diag(sigma2)), \
               NonLazyTensor(torch.diag(sigma2_inv))

    def get_beta(self, size):
        beta_ = nn.Softplus()(self.beta)
        if self.beta is not None:
            beta = torch.diag(beta_)
            beta_inv = torch.diag(beta_.pow(-1))
        return NonLazyTensor(beta), NonLazyTensor(beta_inv)
