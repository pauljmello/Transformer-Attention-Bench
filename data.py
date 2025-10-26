import os
import pickle
import tarfile
import urllib.request

import numpy as np
import torch
from torch.utils.data import TensorDataset


def download_cifar10():
    cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar_filename = "cifar-10-python.tar.gz"

    if not os.path.exists("cifar-10-batches-py"):
        urllib.request.urlretrieve(cifar_url, cifar_filename)
        with tarfile.open(cifar_filename, 'r:gz') as tar:
            tar.extractall('.')
        os.remove(cifar_filename)
        print("CIFAR-10 dataset downloaded.")
    else:
        print("CIFAR-10 dataset found.")


def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')

    X = batch[b'data']
    y = np.array(batch[b'labels'])

    X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return X, y


def normalize_data(X):
    mean = np.mean(X, axis=(0, 1, 2), keepdims=True)
    std = np.std(X, axis=(0, 1, 2), keepdims=True)
    X_normalized = (X - mean) / (std + 1e-8)
    return X_normalized


def load_cifar10():
    download_cifar10()

    data_dir = 'cifar-10-batches-py'

    X_train_list = []
    y_train_list = []

    for i in range(1, 6):
        file_path = f'{data_dir}/data_batch_{i}'
        X_batch, y_batch = load_cifar10_batch(file_path)
        X_train_list.append(X_batch)
        y_train_list.append(y_batch)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    X_test, y_test = load_cifar10_batch(f'{data_dir}/test_batch')

    X_train = normalize_data(X_train.astype(np.float32))
    X_test = normalize_data(X_test.astype(np.float32))

    print(f"CIFAR-10 loaded: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")

    return X_train, y_train, X_test, y_test


def _numpy_to_torch_chw(X_np: np.ndarray) -> torch.Tensor:
    X_t = torch.from_numpy(X_np).permute(0, 3, 1, 2).contiguous()
    return X_t.to(memory_format=torch.channels_last)


def _labels_to_long(y_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(y_np).long()


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(dataset)

        if self.drop_last:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

        if not self.shuffle:
            self.indices = torch.arange(self.num_samples, dtype=torch.long)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples, dtype=torch.long)
        else:
            indices = self.indices

        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)

            batch_indices = indices[start_idx:end_idx]
            batch_data = self.dataset.tensors[0][batch_indices]
            batch_labels = self.dataset.tensors[1][batch_indices]

            yield batch_data, batch_labels


def load_dataset(batch_size: int = 512):
    X_train_np, y_train_np, X_test_np, y_test_np = load_cifar10()

    train_X_t = _numpy_to_torch_chw(X_train_np)
    test_X_t = _numpy_to_torch_chw(X_test_np)
    train_y_t = _labels_to_long(y_train_np)
    test_y_t = _labels_to_long(y_test_np)

    train_dataset = TensorDataset(train_X_t, train_y_t)
    test_dataset = TensorDataset(test_X_t, test_y_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader
