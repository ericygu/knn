import numpy as np
import numpy.testing as npt
import time

def gen_random_samples():

    """
    Generate 5 million random samples using the
    numpy random.randn module.

    Returns
    ----------
    sample : 1d array of size 5 million
        An array of 5 million random samples
    """
    big_random_array = np.random.randn(5000000)
    return big_random_array


def sum_squares_for(samples):
    """
    Bunch of numbers --> can use forloop or numpy vector
    Compute the sum of squares using a forloop

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    timeElapse: float
        The time it took to calculate the sum of squares (in seconds)
    """
    start = time.process_time()
    timeElapse = 0
    ss = 0
    #summation of x^2 values from 1 to N
    for i in samples:
        ss += i**2

    timeElapse += time.process_time() - start
    """
    #sum of squares for dispersion: i.e. "sum of squares," variance, and standard deviation as measures of dispersion code
    
    sigma_x_squared = 0
    N = samples.size
    #get true value of summation of x from 1 to N, squared
    for j in samples:
        sigma_x_squared += j
    sigma_x_squared ** 2
    # Sum of squares for dispersion Formula:
    ss -= (sigma_x_squared/N)
    """
    return ss, timeElapse

def sum_squares_np(samples):
    """
    Compute the sum of squares using Numpy's dot module

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    timeElapse: float
        The time it took to calculate the sum of squares (in seconds)
    """
    start = time.process_time()
    timeElapse = 0
    ss = 0
    #np squares turns every integer x in the array into x^2
    squaredsamples = np.square(samples)
    #we can sum this array up
    ss += np.sum(squaredsamples)

    timeElapse += time.process_time() - start
    return ss, timeElapse

""" 
Time [sec] (for loop): 4.1875
Time [sec] (np loop): 0.03125
np Loop is much faster, I have added a main print statement that will show the difference, this one is specifically
4.15625 seconds faster
"""


def main():
    # generate the random samples
    samples = gen_random_samples()
    # call the sum of squares
    ssFor, timeFor = sum_squares_for(samples)
    # call the numpy version
    ssNp, timeNp = sum_squares_np(samples)
    # make sure they're the same value
    npt.assert_almost_equal(ssFor, ssNp, decimal=5)
    # print out the values
    print("Time [sec] (for loop):", timeFor)
    print("Time [sec] (np loop):", timeNp)
    print("Numpy is faster by time[sec]: ", timeFor - timeNp)

if __name__ == "__main__":
    main()
