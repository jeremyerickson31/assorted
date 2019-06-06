
import numpy
from numpy.linalg import cholesky
from matplotlib import pyplot
from scipy.stats import pearsonr

# constants
vars = 2  # will make 2 variables comprised of random numbers
rand_draws = 10000  # number of random draws
mean = 0.0  # standard normal gauss mean
std_dev = 1.0  # standard normal gauss std.dev

# make high and low correlation matrices
correlation_matrix_low = numpy.array([[1.0, 0.3], [0.3, 1.0]])
correlation_matrix_high = numpy.array([[1.0, 0.95], [0.95, 1.0]])

# make variables from random numbers
rands = numpy.random.normal(mean, std_dev, size=(vars, rand_draws))  # this is a 2 x 10000

# apply cholesky decomposition to get upper triangular
upper_cholesky_low_corr = cholesky(correlation_matrix_low)
upper_cholesky_high_corr = cholesky(correlation_matrix_high)

# the product of rands and upper cholesky generates correlated variables
correlated_rands_low_corr = upper_cholesky_low_corr @ rands
correlated_rands_high_corr = upper_cholesky_high_corr @ rands

# numerical result for the correlation. approximately equal to specified correlation
rands_corr_coeff, p_0 = pearsonr(rands[0, :], rands[1, :])
correlated_rands_corr_coeff_low, p_low = pearsonr(correlated_rands_low_corr[0, :],
                                                  correlated_rands_low_corr[1, :])
correlated_rands_corr_coeff_high, p_high = pearsonr(correlated_rands_high_corr[0, :],
                                                    correlated_rands_high_corr[1, :])

# plots
pyplot.subplot(2, 2, 1)
pyplot.scatter(rands[0, :], rands[1, :], s=1, marker="o")
pyplot.xlabel("Corr=0, actual=" + str(round(rands_corr_coeff, 4)))

pyplot.subplot(2, 2, 2)
pyplot.scatter(correlated_rands_low_corr[0, :], correlated_rands_low_corr[1, :],s=1, marker="o")
pyplot.xlabel("Corr=0.3, actual=" + str(round(correlated_rands_corr_coeff_low, 4)))

pyplot.subplot(2, 2, 3)
pyplot.scatter(rands[0, :], rands[1, :], s=1, marker="o")
pyplot.xlabel("Corr=0, actual=" + str(round(rands_corr_coeff, 4)))

pyplot.subplot(2, 2, 4)
pyplot.scatter(correlated_rands_high_corr[0, :], correlated_rands_high_corr[1, :], s=1, marker="o")
pyplot.xlabel("Corr=0.95, actual=" + str(round(correlated_rands_corr_coeff_high, 4)))

pyplot.show()
