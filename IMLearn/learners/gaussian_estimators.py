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
        biased_var : bool, default=True
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
        self.var_ = X.var()  # TODO maybe drop the ddof

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
        coof = -(1/(2*sigma))
        samples_minus_mu_squared = np.sum((X-mu)**2)

        return coof * samples_minus_mu_squared

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
        exponent = math.exp(-0.5*((sample-mu/math.sqrt(sigma))**2))
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

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
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
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)  # along the rows. #TODO: if samples arrive per col than change axis=0
        # no need for ddof as default computes sum(X)/N-1:
        self.cov_ = np.cov(X, rowvar=False)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
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
        self.cov_det_ = np.linalg.det(self.cov_)
        self.inv_cov_ = np.linalg.inv(self.cov_)
        return pdf_on_mat(X)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        delta = X - mu
        return -np.sum(delta @ cov * delta) # <x_1, Ax_1> +...+ <x_n, Ax_n>


    def probability_density_func_multi(self, samples: np.ndarray) -> float:
        """
        Computes the pdf of a Multivariate Guassian Distribution, as defined
        in literature.
        :param mu: Expectation of Gaussian
        :param cov: Covariance of Gaussian
        :param samples: n samples
        :return: the PDF of a multivariate gaussian sample
        """
        # TODO: how do the samples arrive? each column is a set of samples,
        #  or each row is?
        coof = 1/math.sqrt(((math.pi * 2) ** len(self.cov_)) * np.linalg.det(self.cov_))
        delta = samples-self.mu_
        exponent = math.exp(-0.5*(np.transpose(delta) @ self.inv_cov_ @ delta))
        return coof * exponent
