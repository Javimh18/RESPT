from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from scipy.stats import zscore

DISTANCE_BWTN_PEAKS = 40
F_SAMPLING = 128
BPM_2_1HZ = 60

def stretch_squeeze(x, factor=1):
    """ Uses a exponential function to enlarge the large values in order to find peaks.

        The ecg signal that is passed as argument is normalized

        Parameters
        ----------
        x : numpy array
            The ECG signal sample, obtained from the dataset normalized
        factor: integer
            The factor that goes to the exponent of the exponential function

        Returns
        ------
        The result of the original function stretched
    """
    
    return np.sign(x) * np.exp(factor * np.abs(x))

def computing_rr(x):
    """ Computes the R-R distance between heartbeats. In order to have more information
        About R peaks, visit the following link: https://www.apcollege.edu.au/blog/rr-interval-ecg/

        This function will take the parameter x as the ecg signal. It will apply some transformation
        to enlarge the peaks of the given function and computes thye RR peak distance between them.

        Parameters
        ----------
        x : numpy array
            The ECG signal sample, obtained from the dataset normalized.

        Returns
        ------
        rr: An array with the difference between each peak found in the ecg.
    """

    # Stretching the ecg peaks
    x_st_sq = stretch_squeeze(x)
    
    # finding peaks
    peaks_idx, _ = find_peaks(x_st_sq, distance=DISTANCE_BWTN_PEAKS, height=np.max(x_st_sq[:])*0.2)
    if peaks_idx.size == 0: # some functions does not have positive peaks, but negative ones, we invert the function in order to search for them
        x_st_sq = stretch_squeeze(-x)
        peaks_idx, _ = find_peaks(x_st_sq, distance=DISTANCE_BWTN_PEAKS, height=np.max(x_st_sq)*0.2)
    rr = np.diff(peaks_idx)

    # there are some samples that are quite faulty and we only compute one peak, for those cases
    # we return -1 and we replace the value with the mean of the class in order to not affect the
    # distribution of the data
    if rr.size == 0:
        return -1
    
    # compute the zscore in order to leave out the outliers that maybe product of an anomaly in the data
    rr_pd = pd.Series(rr)
    rr_corrected = rr_pd.copy()
    rr_corrected[np.abs(zscore(rr)) > 2] = np.median(rr)

    rr = np.array(rr_corrected)
    return rr

def avg_d_between_peaks(x):
    """ Computes the mean R-R distance between heartbeats. 

        This function will take the parameter x as the ecg signal and will average the 
        R-R distance in order to obtain a mean rr - distance

        Parameters
        ----------
        x : numpy array
            The ECG signal sample, obtained from the dataset normalized.

        Returns
        ------
        rr: An array with the difference between each peak found in the ecg.
    """

    rr = computing_rr(x)
    rr_pd = pd.Series(rr)

    # If the rr value is just -1, then we return nan
    if rr_pd[rr_pd == -1].any():
        return np.nan
    else:
        return np.mean(rr)
    
def std_d_between_peaks(x):
    """ Computes the std deviation of the R-R distance between heartbeats. 

        This function will take the parameter x as the ecg signal and will compute 
        the standard deviation of the R-R distance.

        Parameters
        ----------
        x : numpy array
            The ECG signal sample, obtained from the dataset normalized.

        Returns
        ------
        std_rr: The std deviation of the rr distance for all r peaks found in the sample x
    """

    rr = computing_rr(x)
    rr_pd = pd.Series(rr)

    # If the rr value is just -1, then we return nan
    if rr_pd[rr_pd == -1].any():
        return np.nan
    else:
        return np.std(rr)

def compute_bpm(x):
    """ Computes the mean of the bpms

        This function will take the parameter x as the ecg signal. It will compute the 
        mean of the bpms found in the sample using the rr distance.

        Parameters
        ----------
        x : numpy array
            The ECG signal sample, obtained from the dataset normalized.

        Returns
        ------
        bpm: Mean bpms for a given signal.
    """

    rr = computing_rr(x)
    rr_pd = pd.Series(rr)

    # If the rr value is just -1, then we return nan
    if rr_pd[rr_pd == -1].any():
        return np.nan
    beat_period = np.mean(rr)/F_SAMPLING
    bpm = BPM_2_1HZ/beat_period
    return bpm

def compute_std_bpm(x):
    """ Computes the standard deviation of the bpms

        This function will take the parameter x as the ecg signal. It will compute the 
        standard deviation of the bpms found in the sample using the rr distance.

        Parameters
        ----------
        x : numpy array
            The ECG signal sample, obtained from the dataset normalized.

        Returns
        ------
        std_bpm: The standard deviation value of the bpms computed using the rr distance.
    """

    rr = computing_rr(x)
    rr_pd = pd.Series(rr)

    # If the rr value is just -1, then we return nan
    if rr_pd[rr_pd == -1].any():
        return np.nan
    beat_period = rr/F_SAMPLING
    std_bpm = np.std(BPM_2_1HZ/beat_period)
    return std_bpm

def compute_max_bpm(x):
    """ Computes the max of the bpms

        This function will take the parameter x as the ecg signal. It will compute the 
        maximum bpm found in the sample using the rr distance.

        Parameters
        ----------
        x : numpy array
            The ECG signal sample, obtained from the dataset normalized.

        Returns
        ------
        max_bpm: The max value of the bpms computed using the rr distance.
    """

    rr = computing_rr(x)
    rr_pd = pd.Series(rr)

    # If the rr value is just -1, then we return nan
    if rr_pd[rr_pd == -1].any():
        return np.nan
    beat_period = rr/F_SAMPLING
    max_bpm = np.max(BPM_2_1HZ/beat_period)
    return max_bpm

def compute_min_bpm(x):
    """ Computes the min of the bpms

        This function will take the parameter x as the ecg signal. It will compute the 
        minimum bpm found in the sample using the rr distance.

        Parameters
        ----------
        x : numpy array
            The ECG signal sample, obtained from the dataset normalized.

        Returns
        ------
        min_bpm: The min value of the bpms computed using the rr distance.
    """

    rr = computing_rr(x)
    rr_pd = pd.Series(rr)

    # If the rr value is just -1, then we return nan
    if rr_pd[rr_pd == -1].any():
        return np.nan
    beat_period = rr/F_SAMPLING
    min_bpm = np.min(BPM_2_1HZ/beat_period)
    return min_bpm