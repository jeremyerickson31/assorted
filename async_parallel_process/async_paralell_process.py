#  will demonstrate the usage of pythons parallel processing capabilities

#  matrix of standard normal values, rows are sim runs and columns are standard normal random variables
#  the sum of each row will make a sum of standard normals
#  percentiles down the sum row should still make a standard normal (mean = 0 + mean = 0 + .... == mean of 0)
#  function will shift each cell by 1 and add the values in the row
#  adding the values is like adding a normal to a normal X number of times
#  this should make a new distribution with mean of N

import numpy
from datetime import datetime

# make variables from random numbers
mean = 0.0  # standard normal mean
std_dev = 1.0  # standard normal standard deviation
sim_runs = 1000000  # number of rows to have; each row is a simulation run
num_vars = 100  # number of columns; each column represents 1 standard normal variable
rands = numpy.random.normal(mean, std_dev, size=(sim_runs, num_vars))
shift_amount = 0  # the amount by which to shift the mean
expected_mean = (0.0 + shift_amount) * num_vars  # = (standard normal mean + shift amount) times how many variables


def add_and_sum(row, add_amount):
    """
    this function will simply add the 'add_amount' to each value in the row and sum the result
    when applied to a standard normal variable it effectively shifts the distribution
    :param row: a row of values
    :param add_amount: integer to add to each value in the row
    :return: the sum of the shifted values
    """

    total = 0.0
    for value in row:
        total += value + add_amount

    return total


#  Run by brute force method of looping over every cell in matrix one at a time
brute_force_sums = list()
brute_strength_start = datetime.now()
for row in rands:
    output = add_and_sum(row, 0)
    brute_force_sums.append(output)
brute_strength_end = datetime.now()

sums_array = numpy.array(brute_force_sums)
print("-------------------")
print("------RESULTS------")
print("-------------------")
print("Method: Brute Force")
print("Excpected Mean: " + str(expected_mean))
print("Actual Mean: " + str(round(sums_array.mean(),5)))
print("Start: " + str(brute_strength_start))
print("End: " + str(brute_strength_end))
print("Duration: " + str(brute_strength_end - brute_strength_start) + "(hrs:mins:sec.msec)")
