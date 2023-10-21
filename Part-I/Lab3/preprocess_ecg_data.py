from scipy.io import loadmat
import pandas as pd
import numpy as np

ECG_LEN = 65536
ECG_WINDOW = 2000
CLASS_SIZES = [96,30,36]

def load_ecg_data(path='./ECGData.mat'):
    matdata = loadmat(path)
    ecg_raw_data = np.array(matdata['ECGData'])[0][0][0]
    # the  values of .mat files are read like a stream (don't take into account each registers)
    # so we splitted said stream into chunks of 65536 values
    ecg_original_data = [ecg_raw_data[i:i + ECG_LEN] for i in range(0, len(ecg_raw_data), ECG_LEN)][0]

    buffer_ecg = []
    for ecg in ecg_original_data:
        ecg_split_list = [ecg[i:i + ECG_WINDOW] for i in range(0, len(ecg), ECG_WINDOW)]
        _ = ecg_split_list.pop() # we need to drop some samples because the last slice of the window takes 1536 values instead of 2000
        buffer_ecg.append(ecg_split_list)

    # reshaping the data, so it fits the 2000 items per window constraint
    ecg_shape = np.array(buffer_ecg).shape
    ecg_data = np.array(buffer_ecg).reshape(-1,2000)

    # assigning for each new instance produced by the window, their corresponding label
    class_labels = [1] * (CLASS_SIZES[0] * ecg_shape[1]) + [2] * (CLASS_SIZES[1] * ecg_shape[1]) + [3] * (CLASS_SIZES[2] * ecg_shape[1])

    # assigning for each sample of each class, their original sample (when there were 162)
    # and the chunk index of their original sample.
    sampled_from = []
    n_chunk = []
    for c_s in CLASS_SIZES:
        for i in range(1, c_s + 1):
            sampled_from.append([i] * ecg_shape[1])
            n_chunk.append(range(1,ecg_shape[1]+1))
        
    sampled_from = np.array(sampled_from).flatten()
    n_chunk = np.array(n_chunk).flatten()

    ecg_labeled_data = [{"ecg_data": ecg_data, "class": label, "sampled_from":sample, "n_chunk":chunk} 
                        for ecg_data, label, sample, chunk in zip(ecg_data, class_labels, sampled_from, n_chunk)]

    return ecg_labeled_data