from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from scipy.stats import zscore

DISTANCE_BWTN_PEAKS = 40
F_SAMPLING = 128
BPM_2_1HZ = 60

def stretch_squeeze(x, factor=1):
    return np.sign(x) * np.exp(factor * np.abs(x))

def computing_rr(x):
    x_st_sq = stretch_squeeze(x)
    # compute cross correlation between ecg and the sine filter
    peaks_idx, _ = find_peaks(x_st_sq, distance=DISTANCE_BWTN_PEAKS, height=np.max(x_st_sq[:])*0.2)
    if peaks_idx.size == 0:
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
    rr = computing_rr(x)
    rr_pd = pd.Series(rr)
    if rr_pd[rr_pd == -1].any():
        return np.nan
    else:
        return np.mean(rr)
    
def std_d_between_peaks(x):
    rr = computing_rr(x)
    rr_pd = pd.Series(rr)
    if rr_pd[rr_pd == -1].any():
        return np.nan
    else:
        return np.std(rr)

def compute_bpm(x):
    rr = computing_rr(x)
    rr_pd = pd.Series(rr)
    if rr_pd[rr_pd == -1].any():
        return np.nan
    beat_period = np.mean(rr)/F_SAMPLING
    bpm = BPM_2_1HZ/beat_period
    return bpm

def compute_std_bpm(x):
    rr = computing_rr(x)
    rr_pd = pd.Series(rr)
    if rr_pd[rr_pd == -1].any():
        return np.nan
    beat_period = rr/F_SAMPLING
    std_bpm = np.std(BPM_2_1HZ/beat_period)
    return std_bpm

def compute_max_bpm(x):
    rr = computing_rr(x)
    rr_pd = pd.Series(rr)
    if rr_pd[rr_pd == -1].any():
        return np.nan
    beat_period = rr/F_SAMPLING
    std_bpm = np.max(BPM_2_1HZ/beat_period)
    return std_bpm

def compute_min_bpm(x):
    rr = computing_rr(x)
    rr_pd = pd.Series(rr)
    if rr_pd[rr_pd == -1].any():
        return np.nan
    beat_period = rr/F_SAMPLING
    std_bpm = np.min(BPM_2_1HZ/beat_period)
    return std_bpm