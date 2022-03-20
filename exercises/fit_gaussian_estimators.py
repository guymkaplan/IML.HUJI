from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"
MU_UNIVARIATE = 10
SCALE_UNIVARIATE = 1
NUM_OF_SAMPLES = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni_gaussian = UnivariateGaussian()
    drawn_samples = np.random.normal(MU_UNIVARIATE, SCALE_UNIVARIATE,
                                     NUM_OF_SAMPLES)
    uni_gaussian.fit(drawn_samples)
    print((uni_gaussian.mu_, uni_gaussian.var_))

    sample_sizes = np.arange(10, 1010, 10)

    calc_distances_from_mu = np.vectorize(lambda size: np.abs(uni_gaussian.fit(drawn_samples[:size]).mu_ - MU_UNIVARIATE))  # TODO figure this out
    distances_from_mu = calc_distances_from_mu(sample_sizes)
    go.Figure([go.Scatter(x=sample_sizes, y=distances_from_mu,
                          mode='markers+lines')], layout=go.Layout(
        title="Absolute Distance Between the Estimated and True Value of the Expectation, as a Function of the Sample Size"
        , xaxis_title="Number of Samples",
        yaxis_title="Distance from Expectation")).show()
    # Question 3 - Plotting Empirical PDF of fitted model
    # I expect to see the Gaussian distribution - a "bell" with it's zenith at
    # the estimated value - mu hat
    samples_pdf = uni_gaussian.pdf(drawn_samples)
    go.Figure([go.Scatter(x=drawn_samples, y=samples_pdf,
                          mode='markers')], layout=go.Layout(
        title=r"Sample values and corresponding PDFS"
        , xaxis_title="Sample Values",
        yaxis_title="PDF")).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    drawn_samples = np.random.multivariate_normal(mean, cov, 1000)
    multi_gaussian = MultivariateGaussian()
    multi_gaussian.fit(drawn_samples)
    print(multi_gaussian.mu_)
    print(multi_gaussian.cov_)


    # Question 5 - Likelihood evaluation
    X_axis = np.linspace(-10, 10, 200)
    theoretical_distribution_x = multi_gaussian.pdf(drawn_samples)

    log_likelihood = np.ndarray((200,200))
    for i in range(200):
        for j in range(200):
            log_likelihood[i, j] = multi_gaussian.log_likelihood(np.array([X_axis[i], 0, X_axis[j], 0]), cov, drawn_samples)
    print(log_likelihood)
    print(1)


    # Question 6 - Maximum likelihood



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
