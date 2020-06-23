#  will demonstrate the usage of pythons parallel processing capabilities

#  matrix of standard normal values, columns are sim runs and rows are standard normal random variables
#  the average of the sim runs is computed for each variable (row)


import numpy
import multiprocessing
from datetime import datetime


def take_average(row):
    """
    this function will simply average the values in each row
    it is used to demonstrate a function that performs a calculation and requires CPU usage
    :param row: a row of values
    :return: the average of the values
    """

    total = 0.0
    for value in row:
        total += value
    average = total / len(row)

    return average


def run_brute_force(array):
    # function to loop through rows of array and apply take_average() to entries in the row
    brute_force_avgs = list()
    brute_force_start = datetime.now()
    for row in array:
        output = take_average(row)
        brute_force_avgs.append(output)
    brute_force_end = datetime.now()
    brute_force_runtime = brute_force_end - brute_force_start

    return brute_force_avgs, brute_force_runtime


def run_parallel(array):
    # function to apply take_average() to each row of an array using parallel processing
    parallel_start = datetime.now()
    proc_pool = multiprocessing.Pool(6)
    output = [proc_pool.apply_async(take_average, args=(row,)) for row in array]
    parallel_avgs = [result.get() for result in output]
    proc_pool.close()
    parallel_end = datetime.now()
    parallel_runtime = parallel_end - parallel_start

    return parallel_avgs, parallel_runtime


def main():
    # make variables from random numbers
    mean = 0.0  # standard normal mean
    std_dev = 1.0  # standard normal standard deviation
    min_num = 1000
    max_num = 2500
    num_rows = max_num  # number of rows; each row represents 1 standard normal variable
    sim_runs = max_num  # number of columns to have; each column is a simulation run with a random draw
    rands = numpy.random.normal(mean, std_dev, size=(num_rows, sim_runs))

    increments = list(range(min_num, max_num, 500))
    results = {"rows": {"brute": [], "parallel": []},
               "columns": {"brute": [], "parallel": []}
               }

    for row_slice in increments:
        # run loop with rows increasing and columns held constant
        print("Running with rows up to " + str(row_slice) + " and all columns")
        rands_slice = rands[0:row_slice + 1, :]
        brute_force_avgs, brute_force_time = run_brute_force(rands)
        parallel_avgs, parallel_time = run_parallel(rands_slice)
        results["rows"]["brute"].append(brute_force_time.total_seconds())
        results["rows"]["parallel"].append(parallel_time.total_seconds())

    for col_slice in increments:
        # run loop with rows held constant and columns increasing
        print("Running with columns up to " + str(col_slice) + " and all rows")
        rands_slice = rands[:, 0:col_slice + 1]
        brute_force_avgs, brute_force_time = run_brute_force(rands)
        parallel_avgs, parallel_time = run_parallel(rands_slice)
        results["columns"]["brute"].append(brute_force_time.total_seconds())
        results["columns"]["parallel"].append(parallel_time.total_seconds())

    print(results)

"""
    print("-------------------")
    print("------RESULTS------")
    print("-------------------")
    print("Brute Force Duration " + str(brute_force_time) + " (hrs:mins:sec.msec)")
    print("Parallel Duration " + str(parallel_time) + " (hrs:mins:sec.msec)")
    print("-------------------")
"""


if __name__ == "__main__":
    main()
