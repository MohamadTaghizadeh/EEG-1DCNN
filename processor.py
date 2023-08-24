import numpy as np
import os
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.datasets import eegdata
from mne.epochs import Epochs
from algorithm1 import compute_features
import mne
from typing import List
import wget
import sys
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
from mne.io import BaseRaw

"""
Algorithm1

"""
channels = [["FC1", "FC2"],
            ["FC3", "FC4"],
            ["FC5", "FC6"],
            ["C5", "C6"],
            ["C3", "C4"],
            ["C1", "C2"],
            ["CP1", "CP2"],
            ["CP3", "CP4"],
            ["CP5", "CP6"]]


class Utils:
  
    combinations = {"a": [["FC1", "FC2"],
                          ["FC3", "FC4"],
                          ["FC5", "FC6"]],

                    "b": [["C5", "C6"],
                          ["C3", "C4"],
                          ["C1", "C2"]],

                    "c": [["CP1", "CP2"],
                          ["CP3", "CP4"],
                          ["CP5", "CP6"]],

                    "d": [["FC3", "FC4"],
                          ["C5", "C6"],
                          ["C3", "C4"],
                          ["C1", "C2"],
                          ["CP3", "CP4"]],

                    "e": [["FC1", "FC2"],
                          ["FC3", "FC4"],
                          ["C3", "C4"],
                          ["C1", "C2"],
                          ["CP1", "CP2"],
                          ["CP3", "CP4"]],

                    "f": [["FC1", "FC2"],
                          ["FC3", "FC4"],
                          ["FC5", "FC6"],
                          ["C5", "C6"],
                          ["C3", "C4"],
                          ["C1", "C2"],
                          ["CP1", "CP2"],
                          ["CP3", "CP4"],
                          ["CP5", "CP6"]]}

    @staticmethod
    def load_data(subjects: List, runs: List, data_path: str) -> List[List[BaseRaw]]:

        all_subject_list = []
        subjects = [str(s) for s in subjects]
        runs = [str(r) for r in runs]
        task2 = [4, 8, 12]
        task4 = [6, 10, 14]
        for sub in subjects:
            if len(sub) == 1:
                sub_name = "S"+"00"+sub
            elif len(sub) == 2:
                sub_name = "S"+"0"+sub
            else:
                sub_name = "S"+sub
            sub_folder = os.path.join(data_path, sub_name)
            single_subject_run = []
            for run in runs:
                if len(run) == 1:
                    path_run = os.path.join(sub_folder, sub_name+"R"+"0"+run+".edf")
                else:
                    path_run = os.path.join(sub_folder, sub_name+"R"+ run +".edf")
                raw_run = read_raw_edf(path_run, preload=True)
                len_run = np.sum(raw_run._annotations.duration)
                if len_run > 124:
                    print(sub)
                    raw_run.crop(tmax=124)

                
                if int(run) in task2:
                    for index, an in enumerate(raw_run.annotations.description):
                        if an == "T0":
                            raw_run.annotations.description[index] = "B"
                        if an == "T1":
                            raw_run.annotations.description[index] = "L"
                        if an == "T2":
                            raw_run.annotations.description[index] = "R"
                if int(run) in task4:
                    for index, an in enumerate(raw_run.annotations.description):
                        if an == "T0":
                            raw_run.annotations.description[index] = "B"
                        if an == "T1":
                            raw_run.annotations.description[index] = "LR"
                        if an == "T2":
                            raw_run.annotations.description[index] = "F"
                single_subject_run.append(raw_run)
            all_subject_list.append(single_subject_run)
        return all_subject_list

    @staticmethod
    def eeg_settings(raws:  List[BaseRaw]) -> List[BaseRaw]:
        
        raw_setted = []
        for subj in raws:
            eegdata.standardize(subj)
            montage = make_standard_montage('standard_1005')
            subj.set_montage(montage)
            raw_setted.append(subj)

        return raw_setted


    @staticmethod
    def select_channels(raws: List[BaseRaw], ch_list: List ) -> List[BaseRaw]:
        
        s_list = []
        for raw in raws:
            s_list.append(raw.pick_channels(ch_list))

        return s_list

    @staticmethod
    def epoch(raws: List[BaseRaw], exclude_base: bool =False,
              tmin: int =0, tmax: int =4) -> (np.ndarray, List):
        
        xs = list()
        ys = list()
        for raw in raws:
            if exclude_base:
                event_id = dict(F=2, L=3, LR=4, R=5)
            else:
                event_id = dict(B=1, F=2, L=3, LR=4, R=5)
            tmin, tmax = tmin, tmax
            events, _ = mne.events_from_annotations(raw, event_id=event_id)

            picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                                   exclude='bads')
            epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)

            y = list()
            for index, data in enumerate(epochs):
                y.append(epochs[index]._name)

            xs.append(np.array([epoch for epoch in epochs]))
            ys.append(y)

        return np.concatenate(tuple(xs), axis=0), [item for sublist in ys for item in sublist]

    @staticmethod
    def cut_width(data):
        new_data = np.zeros((data.shape[0], data.shape[1], data.shape[2] - 1))
        for index, line in enumerate(data):
            new_data[index] = line[:, : -1]

        return new_data

    @staticmethod
    def load_sub_by_sub(subjects, data_path, name_single_sub):
        xs = list()
        ys = list()
        for sub in subjects:
            xs.append(Utils.cut_width(np.load(os.path.join(data_path, "x" + name_single_sub + str(sub) + ".npy"))))
            ys.append(np.load(os.path.join(data_path, "y" + name_single_sub + str(sub) + ".npy")))
        return xs, ys


    @staticmethod
    def scale_sub_by_sub(xs, ys):
        for sub_x, sub_y, sub_index in zip(xs, ys, range(len(xs))):
            for sample_index, x_data in zip(range(sub_x.shape[0]), sub_x):
                xs[sub_index][sample_index] = minmax_scale(x_data, axis=1)

        return xs, ys


    @staticmethod
    def to_one_hot(y, by_sub=False):
        if by_sub:
            new_array = np.array(["nan" for nan in range(len(y))])
            for index, label in enumerate(y):
                new_array[index] = ''.join([i for i in label if not i.isdigit()])
        else:
            new_array = y.copy()
        total_labels = np.unique(new_array)
        mapping = {}
        for x in range(len(total_labels)):
            mapping[total_labels[x]] = x
        for x in range(len(new_array)):
            new_array[x] = mapping[new_array[x]]

        return tf.keras.utils.to_categorical(new_array)


    @staticmethod
    def train_test_split(x, y, perc):
        from numpy.random import default_rng
        rng = default_rng()
        test_x = list()
        train_x = list()
        train_y = list()
        test_y = list()
        for sub_x, sub_y in zip(x, y):
            how_many = int(len(sub_x) * perc)
            indexes = np.arange(0, len(sub_x))
            choices = rng.choice(indexes, how_many, replace=False)
            for sample_x, sample_y, index in zip(sub_x, sub_y, range(len(sub_x))):
                if index in choices:
                    test_x.append(sub_x[index])
                    test_y.append(sub_y[index])
                else:
                    train_x.append(sub_x[index])
                    train_y.append(sub_y[index])
        return np.dstack(tuple(train_x)), np.dstack(tuple(test_x)), np.array(train_y), np.array(test_y)

    @staticmethod
    def load(channels, subjects, base_path):
        data_x = list()
        data_y = list()

        for couple in channels:
            data_path = os.path.join(base_path, couple[0] + couple[1])
            sub_name = "_sub_"
            xs, ys = Utils.load_sub_by_sub(subjects, data_path, sub_name)
            data_x.append(np.concatenate(xs))
            data_y.append(np.concatenate(ys))

        return np.concatenate(data_x), np.concatenate(data_y)

if __name__ == "__main__":
    pass
