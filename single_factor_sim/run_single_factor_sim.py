
# This script will give an example of the Single Factor Asset Correlation model
# The Single Factor model has the form R_ij = sqrt(rho_i) * Z_i + sqrt(1-rho_i) * epsilon_ij
# Then, with a known default probability p_j, the loan/firm defaults if R_ij < inverse_std_norm(p_j)
# This script will generate a random portfolio of credit exposures each with an amount, loss and default probability
# The script will then run the simulation to generate a portfolio loss distribution
# The goal will be show descriptive statistics on the loss distribution when the asset correlation value is changed
# For comparison purposes this will also generate the loss distribution percentiles from the Vasicek formula:
# SUM( Weight_i * LGD_i * norm((norm_inv(PD_i) + (sqrt(corr_i) * norm_inv(alpha))) / sqrt(1.0 - corr_i))
# below we use the same correlation for each loan in the pool

# imports
import math
import numpy
import pandas
import random
import scipy.stats

from matplotlib import pyplot

# constants
mu = 0.0  # for standard norm dist
sigma = 1.0  # for standard norm dist
corr = 0.05  # asset correlations to use for all loans in sim
pctls = numpy.linspace(0.001, 1.00, 999, False)  # list of percentiles for Vasicek calc
num_loans = 1000  # number of loans in the pretend pool
sim_runs = 10000  # number of simulation runs to do
loan_bal_min = 10000  # minimum loan balance 10,000
loan_bal_max = 10000000  # maximum loan balance 10,000,000

# pre-make the standard normal functions
norm = scipy.stats.norm(mu, sigma).cdf  # cumulative standard normal
norm_inv = scipy.stats.norm(mu, sigma).ppf  # inverse cumulative standard normal

# generate a pool of loans that have random PDs, LGDs and Balances
loan_pds = [random.random() for i in range(0, num_loans)]  # make loan level default probs
loan_lgds = [random.random() for i in range(0, num_loans)]  # make loan level loss given defaults
loan_bals = [random.randrange(loan_bal_min, loan_bal_max, 1000) for i in range(0, num_loans)]  # make loan balances

# pre-make the random draws for Z and epsilon. these can be re-used to run calculation on exact same set of random draws
z_vector_static = numpy.random.normal(loc=mu, scale=sigma, size=(1, sim_runs))  # vector of randoms for Z_i
epsilon_matrix_static = numpy.random.normal(loc=mu, scale=sigma, size=(num_loans, sim_runs))  # matrix of randoms for e_ij


def get_vasicek_dist(pds, lgds, bals):
    """
    calculates the percentile of the loss distribution as given by the Vasicek equation
    :param pds: list of default probabilities
    :param lgds: list of loss given defaults
    :param bals: list of loan balances
    :return:
    """
    # make DataFrame from input lists
    pool = pandas.DataFrame([bals, pds, lgds], index=["Balance", "PD", "LGD"])
    pool = pool.transpose()  # index 1,2,3..., columns Balance, PD, LGD
    pool["Weight"] = pool["Balance"] / pool["Balance"].sum()  # Weight column is % of Balance for pool

    # take each percentile and evaluate Vasicek equation
    vas_loss_dist = list()
    for alpha in pctls:
        # for each alpha in list of percentiles, calculate the loss % by using the Vasicek equation
        pool["pct_" + str(alpha)] = pool["Weight"] * \
                                    pool["LGD"] * \
                                    norm((norm_inv(pool["PD"]) + (math.sqrt(corr) * norm_inv(alpha))) / math.sqrt(1.0 - corr))
        vas_loss_dist.append(pool["pct_" + str(alpha)].sum())

    return vas_loss_dist


def brute_force_sim(pds, lgds, bals):
    """
    performs the asset correlation simulation with brute force double for-loop
    :param pds: list of default probabilities
    :param lgds: list of loss given defaults
    :param bals: list of loan balances
    :return:
    """
    # ######### Single Factor Asset Correlation Model Simulation #########
    sim_run_loss_list = list()  # list to hold $ loss on portfolio for each simulation run
    for i in range(0, sim_runs):
        print(i)  # simulation run count

        sim_run_loss = 0.0  # outstanding balance loss for each sim run
        Z_i = norm_inv(random.random())  # random draw on the systematic factor

        for j in range(0, num_loans):
            epsilon_ij = norm_inv(random.random())  # random draw on the idiosyncratic factor
            R_ij = math.sqrt(corr) * Z_i + math.sqrt(1.0 - corr) * epsilon_ij  # calculate single factor asset return

            if R_ij < norm_inv(pds[j]):  # if asset return is less than norm of loan pd
                is_defaulted = True
            else:
                is_defaulted = False

            if is_defaulted:
                loss_amt = lgds[j] * bals[j]
            else:
                loss_amt = 0.0

            sim_run_loss += loss_amt

        sim_run_loss_list.append(sim_run_loss)

    return sim_run_loss_list


def matrix_calc_sim(pds, lgds, bals, z_vec_in=None, epsilon_mat_in=None):
    """
    performs the asset correlation simulation with numpy matrix operations
    :param pds: list of default probabilities
    :param lgds: list of loss given defaults
    :param bals: list of loan balances
    :param z_vec_in: array of pre-drawn Z_i value for the sim, to run calc on exact same random variables
    :param epsilon_mat_in: matrix of pre-drawn epsilon_ij values for the sim, to run calc on exact same random variables
    :return:
    """

    if z_vec_in is None:
        # make new set of randoms for Z_i
        z_vector = numpy.random.normal(loc=mu, scale=sigma, size=(1, sim_runs))  # vector of randoms for Z_i
    else:
        # use the set of randoms you were given
        z_vector = z_vec_in

    if epsilon_mat_in is None:
        # make new set of randoms for epsilon_ij
        epsilon_matrix = numpy.random.normal(loc=mu, scale=sigma, size=(num_loans, sim_runs))  # matrix of randoms for e_ij
    else:
        # use the set of randoms you were given
        epsilon_matrix = epsilon_mat_in

    # makes use of z_vector broadcasting to epsilon matrix size
    r_ij_matrix = (math.sqrt(corr) * z_vector) + (math.sqrt(1.0 - corr) * epsilon_matrix)

    pds_to_inv_norm = [norm_inv(pd) for pd in pds]  # run the pds through the norm inverse
    pds_vector = numpy.array(pds_to_inv_norm).reshape((num_loans, 1))  # makes a column vector of loan level PDs
    loan_loss_vector = (numpy.array(lgds) * numpy.array(bals)).reshape((num_loans, 1))

    is_defaulted_mask = r_ij_matrix < pds_vector  # result is a matrix filled with True and False
    loan_loss_matrix = loan_loss_vector * is_defaulted_mask  # False * number = 0, True * number = number
    sim_run_loss_list = list(loan_loss_matrix.sum(axis=0))  # sum down column is sum for all loans in sim run
    return sim_run_loss_list


if __name__ == "__main__":

    sim_run_loss_list = matrix_calc_sim(loan_pds, loan_lgds, loan_bals, z_vec_in=z_vector_static, epsilon_mat_in=epsilon_matrix_static)
    #sim_run_loss_list = brute_force_sim(loan_pds, loan_lgds, loan_bals)
    sim_loss_frame = pandas.DataFrame(sim_run_loss_list)
    sim_loss_frame.hist(bins=50, grid=True, xrot=90)
    pyplot.show()


