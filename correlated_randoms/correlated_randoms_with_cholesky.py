
import numpy
from numpy.linalg import cholesky
from matplotlib import pyplot
from scipy.stats import pearsonr

vars = 2
rand_draws = 10000
mean = 0.0
std_dev = 1.0

correlation_matrix_low = numpy.array([[1.0, 0.3], [0.3, 1.0]])
correlation_matrix_high = numpy.array([[1.0, 0.9], [0.9, 1.0]])
rands = numpy.random.normal(mean, std_dev, size=(vars, rand_draws))
upper_cholesky_low_corr = cholesky(correlation_matrix_low)
upper_cholesky_high_corr = cholesky(correlation_matrix_high)
correlated_rands_low_corr = upper_cholesky_low_corr @ rands
correlated_rands_high_corr = upper_cholesky_high_corr @ rands

rands_corr_coeff, p_0 = pearsonr(rands[0, :], rands[1, :])
correlated_rands_corr_coeff_low, p_low = pearsonr(correlated_rands_low_corr[0, :],
                                                  correlated_rands_low_corr[1, :])
correlated_rands_corr_coeff_high, p_high = pearsonr(correlated_rands_high_corr[0, :],
                                                    correlated_rands_high_corr[1, :])

pyplot.subplot(2, 2, 1)
pyplot.scatter(rands[0, :], rands[1, :])
pyplot.subplot(2, 2, 2)
pyplot.scatter(correlated_rands_low_corr[0, :], correlated_rands_low_corr[1, :])
pyplot.subplot(2, 2, 3)
pyplot.scatter(rands[0, :], rands[1, :])
pyplot.subplot(2, 2, 4)
pyplot.scatter(correlated_rands_high_corr[0, :], correlated_rands_high_corr[1, :])

pyplot.show()
