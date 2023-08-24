import mne
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

"""
    Physionet MI-EEG Dataset
    64 channels EEG，160hz freq, 4 seconds MI-task
    14 runs for each of the 109 subjects
        runs [1, 2] is baseline
        others with marker
            T0   : rest, 
            T1/T2: left/right fist in runs [3, 4, 7, 8, 11, 12]
                   both fists/feet in runs [5, 6, 9, 10, 13, 14]

"""
data_path = r'..\S001\\'
LR_fist_run = [3, 4, 7, 8, 11, 12]
fist_feet_run = [5, 6, 9, 10, 13, 14]
rename_mapping = {'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCZ', 'Fc2.': 'FC2', 'Fc4.': 'FC4',
                  'Fc6.': 'FC6', 'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1', 'Cz..': 'CZ', 'C2..': 'C2', 'C4..': 'C4',
                  'C6..': 'C6', 'Cp5.': 'CP5', 'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPZ', 'Cp2.': 'CP2',
                  'Cp4.': 'CP4', 'Cp6.': 'CP6', 'Fp1.': 'FP1', 'Fpz.': 'FPZ', 'Fp2.': 'FP2', 'Af7.': 'AF7',
                  'Af3.': 'AF3', 'Afz.': 'AFZ', 'Af4.': 'AF4', 'Af8.': 'AF8', 'F7..': 'F7', 'F5..': 'F5', 'F3..': 'F3',
                  'F1..': 'F1', 'Fz..': 'FZ', 'F2..': 'F2', 'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8', 'Ft7.': 'FT7',
                  'Ft8.': 'FT8', 'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10', 'Tp7.': 'TP7', 'Tp8.': 'TP8',
                  'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1', 'Pz..': 'PZ', 'P2..': 'P2', 'P4..': 'P4',
                  'P6..': 'P6', 'P8..': 'P8', 'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POZ', 'Po4.': 'PO4', 'Po8.': 'PO8',
                  'O1..': 'O1', 'Oz..': 'OZ', 'O2..': 'O2', 'Iz..': 'IZ'}


def get_physionet(subject: int):
    """
    :param subject: SN of subject : [1,109]
    :return: data shapes (-1, channels, 640)
    """
    # loading from file
    for r in LR_fist_run:
        raw_new = mne.io.read_raw_edf(data_path + 'S%03d' % subject + 'R%02d.edf' % r, verbose='ERROR')
        if r == LR_fist_run[0]:
            raw_LR_fist = raw_new
        else:
            raw_LR_fist.append(raw_new)
    for r in fist_feet_run:
        raw_new = mne.io.read_raw_edf(data_path + 'S%03d' % subject + 'R%02d.edf' % r, verbose='ERROR')
        if r == fist_feet_run[0]:
            raw_fist_feet = raw_new
        else:
            raw_fist_feet.append(raw_new)

    raw_LR_fist.rename_channels(rename_mapping)
    raw_fist_feet.rename_channels(rename_mapping)
    ch_pick = ["FC1", "FC2", "FC3", "FC4", "C3", "C4", "C1", "C2",
               "CP1", "CP2", "CP3", "CP4"]

    # get the data and labels
    event_id_LR_fist = dict(T1=0, T2=1)
    events, _ = mne.events_from_annotations(raw_LR_fist, event_id_LR_fist, verbose='ERROR')
    epochs_LR_fist = mne.Epochs(raw_LR_fist, events, tmin=1 / 160, tmax=4, baseline=None, preload=True,
                                verbose='ERROR')
    event_id_fist_feet = dict(T1=2, T2=3)
    events, _ = mne.events_from_annotations(raw_fist_feet, event_id_fist_feet, verbose='ERROR')
    epochs_fist_feet = mne.Epochs(raw_fist_feet, events, tmin=1 / 160, tmax=4, baseline=None, preload=True,
                                  verbose='ERROR')
    data = np.concatenate((epochs_LR_fist.get_data(picks=ch_pick), epochs_fist_feet.get_data(picks=ch_pick)))
    scaler = preprocessing.StandardScaler()
    for i in range(len(data)):
        scaler.fit(data[i])
        data[i] = scaler.transform(data[i])
    labels = np.concatenate((epochs_LR_fist.events[:, 2], epochs_fist_feet.events[:, 2]))
    labels = to_categorical(labels)  # one-hot

    # reshape and return
    train_data_ori, test_data_ori, train_label_ori, test_label_ori = train_test_split(data, labels, test_size=0.2,
                                                                                      random_state=42)
    train_data = np.empty((0, 2, train_data_ori.shape[2]))
    train_label = np.empty((0, 4))
    test_data = np.empty((0, 2, test_data_ori.shape[2]))
    test_label = np.empty((0, 4))
    for i in range(0, len(ch_pick), 2):
        train_data = np.concatenate((train_data, train_data_ori[:, i:i + 2, :]))
        test_data = np.concatenate((test_data, test_data_ori[:, i:i + 2, :]))
        train_label = np.concatenate((train_label, train_label_ori))
        test_label = np.concatenate((test_label, test_label_ori))
    print('data loaded.')
    return train_data, test_data, train_label, test_label


if __name__ == '__main__':
    res = get_physionet(1)
    for r in res:
        print(r.shape)