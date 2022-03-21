from __future__ import annotations

import math

import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None


    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean()
        # n-1 is for unbiased, n is for biased
        if self.biased_:
            self.var_ = X.var()
        else:
            self.var_ = X.var(ddof=1)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        pdf_on_vector = np.vectorize(self.probability_density_func_uni)
        return pdf_on_vector(self.mu_, self.var_, X)


    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        return -(len(X)/2) * math.log(2 * math.pi * (sigma)) - (np.sum(
            ((X - mu) ** 2)) / (2 * (sigma)))


    @staticmethod
    def probability_density_func_uni(mu: float, sigma: float, sample: float) -> float:
        """
        Computes the pdf of a Univariate Guassian Distribution, as defined
        in literature.
        :param mu: Expectation of Gaussian
        :param sigma: Variance of Gaussian
        :param sample: a single sample
        :return: the PDF of a single sample
        """
        coof = 1/math.sqrt(sigma * 2 * math.pi)
        exponent = math.exp((-1/(2*sigma))*((sample-mu)**2))
        return coof * exponent


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.inv_cov_ = None
        self.cov_det_ = 0

        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        # no need for ddof as default computes sum(X)/N-1:
        self.cov_ = np.cov(X, rowvar=False)
        self.cov_det_ = np.linalg.det(self.cov_)
        self.inv_cov_ = np.linalg.inv(self.cov_)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        pdf_on_mat = np.vectorize(self.probability_density_func_multi)

        return pdf_on_mat(X)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        A = np.linalg.inv(cov)
        detA = np.linalg.det(A)
        coof = -((len(X) * len(A))/2) * np.log(2 * np.pi) - ((len(X)/2) * np.log(detA))
        delta = X - mu
        return coof - (0.5 * np.sum((delta @ A) * delta))  # <x_1, Ax_1> +...+ <x_n, Ax_n>

        # return (len(X) / 2 * (np.log(1 / (1 / np.sqrt(np.linalg.det(cov) * ((2 * math.pi) ** (len(cov)))))))) - \
        #        (0.5 * np.sum(((X - mu) @ (np.linalg.inv(cov))) * (X - mu)))


    def probability_density_func_multi(self, samples: np.ndarray) -> float:
        """
        Computes the pdf of a Multivariate Guassian Distribution, as defined
        in literature.
        :param mu: Expectation of Gaussian
        :param cov: Covariance of Gaussian
        :param samples: n samples
        :return: the PDF of a multivariate gaussian sample
        """
        coof = 1/math.sqrt(((math.pi * 2) ** len(self.cov_)) * self.cov_det_)
        delta = samples-self.mu_
        exponent = math.exp(-0.5*(np.transpose(delta) @ self.inv_cov_ @ delta))
        return coof * exponent
