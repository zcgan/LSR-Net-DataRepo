from torch.utils.data import Dataset
import h5py
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import random
random.seed(0)

class CH_GRF_Dataset:
    def __init__(self, path,transform=None):
        h5f_data = h5py.File(path, 'r')
        self.keys = list(h5f_data.keys())  
        self.transform = transform
        self.path = path
        h5f_data.close()


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with h5py.File(self.path, 'r') as h5f:
            key = self.keys[index]
            data = h5f[key][:]

        return data


def train_val_dataset(dataset, seed, test_split=0.2,val_split=0.2):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split, random_state=seed,
                                           shuffle=True)
    datasets = {}
    datasets['train_item'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, test_idx)
    train_idx, val_idx = train_test_split(list(range(len(datasets['train_item']))), test_size=val_split, random_state=seed,
                                          shuffle=True)
    datasets['train'] = Subset(datasets['train_item'], train_idx)
    datasets['val'] = Subset(datasets['train_item'], val_idx)
    return datasets
#