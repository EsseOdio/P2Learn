import random

import torch
import torchvision

from torch.utils.data import DataLoader, Subset


def read_conf(d):
    print("\nReading configuration file...")
    with open('./config.txt') as f:
        for line in f:
            if line != "\n":
                k, v = line.split()
                d[k] = v
    print("done")


def init_weights(model):
    def init_func(mdl):
        classname = mdl.__class__.__name__
        if hasattr(mdl, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            torch.nn.init.xavier_normal_(mdl.weight.data, 1)
            print(classname, "weight initialized")

    model.apply(init_func)


def load_data(data_path, rand_seed, batch_size):
    transform = torchvision.transforms.ToTensor()

    train_dataset = torchvision.datasets.MNIST(root=data_path, download=True, train=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_path, download=True, train=False, transform=transform)

    num_train = len(train_dataset)
    num_test = len(test_dataset)

    train_idx = list(range(num_train))
    test_idx = list(range(num_test))

    random.seed(rand_seed)
    random.shuffle(train_idx)

    # Define loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    # Define dictionary of loaders
    loaders = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_idx": train_idx,
        "test_idx": test_idx
    }

    return loaders
