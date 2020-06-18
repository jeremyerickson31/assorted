#  will demonstrate the usage of pythons parallel processing capabilities

#  matrix of standard normal values, columns are sim runs and rows are standard normal random variables
#  the average of the sim runs is computed for each variable (row)


import numpy
from datetime import datetime

# make variables from random numbers
mean = 0.0  # standard normal mean
std_dev = 1.0  # standard normal standard deviation
num_rows = 1000  # number of rows; each row represents 1 standard normal variable
sim_runs = 1000  # number of columns to have; each column is a simulation run with a random draw
rands = numpy.random.normal(mean, std_dev, size=(num_rows, sim_runs))


def take_average(row):
    """
    this function will simply average the values in each row
    it is used to demostrate a function that performs a calculation and requires CPU usage
    :param row: a row of values
    :return: the average of the values
    """

    total = 0.0
    for value in row:
        total += value
    average = total / len(row)

    return average


#  Run by brute force method of looping over every cell in matrix one at a time
brute_force_avgs = list()
brute_force_start = datetime.now()
for row in rands:
    output = take_average(row)
    brute_force_avgs.append(output)
brute_strength_end = datetime.now()

print("-------------------")
print("------RESULTS------")
print("-------------------")
print("Method: Brute Force")
print("Start: " + str(brute_force_start))
print("End: " + str(brute_strength_end))
print("Duration: " + str(brute_strength_end - brute_force_start) + " (hrs:mins:sec.msec)")
print("-------------------")
