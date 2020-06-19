#  will demonstrate the usage of pythons parallel processing capabilities

#  matrix of standard normal values, columns are sim runs and rows are standard normal random variables
#  the average of the sim runs is computed for each variable (row)


import numpy
import multiprocessing
from datetime import datetime


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


def run_brute_force(array):
    brute_force_avgs = list()
    for row in array:
        output = take_average(row)
        brute_force_avgs.append(output)
    return brute_force_avgs


def run_parallel(array):
    proc_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    output = [proc_pool.apply_async(take_average, args=(row,)) for row in array]
    parallel_avgs = [result.get() for result in output]
    proc_pool.close()
    return parallel_avgs


def main():
    # make variables from random numbers
    mean = 0.0  # standard normal mean
    std_dev = 1.0  # standard normal standard deviation
    num_rows = 5000  # number of rows; each row represents 1 standard normal variable
    sim_runs = 5000  # number of columns to have; each column is a simulation run with a random draw
    rands = numpy.random.normal(mean, std_dev, size=(num_rows, sim_runs))

    rands_slice = rands

    #  Run by brute force method of looping over every cell in matrix one at a time
    brute_force_start = datetime.now()
    brute_force_avgs = run_brute_force(rands_slice)
    brute_strength_end = datetime.now()

    # run by parallel processing
    parallel_start = datetime.now()
    parallell_avgs = run_parallel(rands_slice)
    parallel_end = datetime.now()

    print("-------------------")
    print("------RESULTS------")
    print("-------------------")
    print("Brute Force Duration " + str(brute_strength_end - brute_force_start) + " (hrs:mins:sec.msec)")
    print("Parallel Duration " + str(parallel_end - parallel_start) + " (hrs:mins:sec.msec)")
    print("-------------------")



if __name__ == "__main__":
    main()
