## dataset_me_new_npy_2.py
import os
import scipy.io as scio
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import platform

class Parsing_Pose_PC(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    {
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h': 8,
    'i': 9,
    'j': 10,
    'k': 11,
    'l': 12,
    'm': 13,
    'n': 14,
    'o': 15,
    'p': 16,
    'q': 17,
    'r': 18,
    's': 19,
    't': 20,
    'u': 21,
    'v': 22,
    'w': 23,
    'x': 24,
    'y': 25,
    'z': 26,
    '0': 27,
    '1': 28,
    '2': 29,
    '3': 30,
    '4': 31,
    '5': 32,
    '6': 33,
    '7': 34,
    '8': 35,
    '9': 36,
}

    """

    def __init__(self, train=True):
        self.Keyboard = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
        self.classes = dict(zip(sorted(self.Keyboard), range(len(self.Keyboard))))
        self.train = train

        if self.train:  # train
            self.data_ti_, self.data_label_, self.data_key2_ = self.dataRead()
            self.data_ti = self.data_ti_
            self.data_label = self.data_label_
            self.data_key2 = self.data_key2_
        else:  # test
            self.data_ti_, self.data_label_, self.data_key2_ = self.dataRead1()
            self.data_test_ti = self.data_ti_
            self.data_test_label = self.data_label_
            self.data_test_key2 = self.data_key2_

    def __getitem__(self, index):
        if self.train:
            ti, label = self.data_ti[index], self.classes[self.data_label[index]]

        else:  # test 固定的ti kinect 数据
            ti, label = self.data_test_ti[index], self.classes[self.data_test_label[index]]

        return ti, label

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):  # train
        # list_all_ti = []  # ti数据
        # list_all_label = []  # body label
        # list_all_key2 = []  # joint  point
        if platform.system() == 'Linux':
            root_dir = "/home/shiruili/"
        else:
            root_dir = r"C:\Users\83723\Downloads\LF-PCT\Point Cloud_add_all"

        list_all_ti = np.load(os.path.join(root_dir,r"npyData/train/list_all_ti.npy"))
        list_all_label = np.load(os.path.join(root_dir,r"npyData/train/list_all_label.npy"))
        list_all_key2 = np.load(os.path.join(root_dir,r"npyData/train/list_all_key2.npy"))

        list_all_ti = np.asarray(list_all_ti)
        list_all_label = np.asarray(list_all_label)
        list_all_key2 = np.asarray(list_all_key2)

        print("train_data load end")
        return list_all_ti, list_all_label, list_all_key2

    def normalization(self, data, key):
        # _range = np.max(data) - np.min(data)
        # return (data - np.min(data)) / _range, (key - np.min(data)) / _range
        return data, key

    def dataRead1(self):
        if platform.system() == 'Linux':
            root_dir = "/home/shiruili/"
        else:
            root_dir = r"C:\Users\83723\Downloads\LF-PCT\Point Cloud_add_all"

        list_all_ti = np.load(os.path.join(root_dir,r"npyData/test/list_all_ti.npy"))
        list_all_label = np.load(os.path.join(root_dir,r"npyData/test/list_all_label.npy"))
        list_all_key2 = np.load(os.path.join(root_dir,r"npyData/test/list_all_key2.npy"))

        print("test_data load end")
        list_all_ti = np.asarray(list_all_ti)
        list_all_label = np.asarray(list_all_label)
        list_all_key2 = np.asarray(list_all_key2)
        return list_all_ti, list_all_label, list_all_key2


if __name__ == '__main__':
    #batch_size个bag,取25mat，每个mat中含有65×5个点
    #batch_size个动作，取25帧，每帧中含有64*1的点
    train_data = Parsing_Pose_PC()
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True, num_workers=5)

    for data in train_loader:
        pcloud,targets=data
    print(pcloud.shape)