from scipy.io import loadmat
import numpy as np

ECG_LEN = 65536
ECG_WINDOW = 2000
CLASS_SIZES = [96,30,36]

def load_ecg_data(path='./ECGData.mat'):
    """ Takes the raw data in matlab format and shapes it in samples of length equal to 2000.

        The function is divided in the following steps:

            * 1. Unpack the data. The loadmat function returns an stream with all the values for all the samples.
            We slice the data in windows of 65536 values, giving is the total 162 samples.
            * 2. Windowing. Once we have the samples, we again, window the data in windows of 2000 values. The last
            window is left out since does not match the 2000 values constrain
            * 3. Assign labels. We then, assign the corresponding labels to each of the windows. We do it by enlarging 
            each label from the original sample by 32 (number of subsamples of length 2000 from the original 65536 length) 
            to match the new number of samples.
            * 4. Adding metadata. At last, we add some additional fields, such as the original sample where the sample was extracted
            and the chunk assigned to the original sample.

        Parameters
        ----------
        path: path to the file where the dat lies.

        Returns
        ------
        ecg_labeled_data: The dataframe with all the samples of 2000 of length
    """
        
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