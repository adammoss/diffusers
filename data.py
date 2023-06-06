import os
import requests
from tqdm import tqdm

import numpy as np
from skimage.transform import resize, downscale_local_mean, resize_local_mean

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, parameters, data_conditional=None, augment=True):
        self.data = torch.from_numpy(data)
        self.parameters = torch.from_numpy(parameters)
        self.augment = augment
        if data_conditional is not None:
            self.data_conditional = torch.from_numpy(data_conditional)
        else:
            self.data_conditional = None

    def __getitem__(self, index):
        x = self.data[index]
        y = self.parameters[index]
        if self.data_conditional is not None:
            x_conditional = self.data_conditional[index]
        else:
            x_conditional = None

        if self.augment:
            if np.random.rand() < 0.5:
                x = torch.flip(x, [1, ])
                if x_conditional is not None:
                    x_conditional = torch.flip(x_conditional, [1, ])
            if np.random.rand() < 0.5:
                x = torch.flip(x, [2, ])
                if x_conditional is not None:
                    x_conditional = torch.flip(x_conditional, [2, ])
            k = np.random.choice([0, 1, 2, 3])
            if k > 0:
                x = torch.rot90(x, k=k, dims=[1, 2])
                if x_conditional is not None:
                    x_conditional = torch.rot90(x_conditional, k=k, dims=[1, 2])

        if x_conditional is not None:
            return {"input": x, "parameters": y, "conditional_input": x_conditional}
        else:
            return {"input": x, "parameters": y}

    def __len__(self):
        return len(self.data)


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def get_cmd_dataset(dataset_name, cache_dir='.', data_size=None, resolution=None, transform=np.log,
                    accelerator=None, norm_min=-1, norm_max=1):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if (accelerator is None or accelerator.is_main_process) and \
            not os.path.isfile(os.path.join(cache_dir, 'Maps_%s_LH_z=0.00.npy' % dataset_name)):
        url = 'https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/Maps_%s_LH_z=0.00.npy' % dataset_name
        download(url, os.path.join(cache_dir, 'Maps_%s_LH_z=0.00.npy' % dataset_name))

    if 'simba' in dataset_name.lower():
        parameter_file = 'params_SIMBA.txt'
    elif 'illustris' in dataset_name.lower():
        parameter_file = 'params_IllustrisTNG.txt'

    if (accelerator is None or accelerator.is_main_process) and \
            not os.path.isfile(os.path.join(cache_dir, parameter_file)):
        url = 'https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/%s' % parameter_file
        download(url, os.path.join(cache_dir, parameter_file))

    if accelerator is not None:
        accelerator.wait_for_everyone()

    Y = np.loadtxt(os.path.join(cache_dir, parameter_file)).astype(np.float32)
    Y = np.repeat(Y, 15, axis=0)
    if data_size is not None:
        Y = Y[0:data_size]
    minimum = np.min(Y, axis=0)
    maximum = np.max(Y, axis=0)
    Y = (Y - minimum) / (maximum - minimum)
    Y = np.expand_dims(Y, 1)

    X = np.load(os.path.join(cache_dir, 'Maps_%s_LH_z=0.00.npy' % dataset_name)).astype(np.float32)
    if data_size is not None:
        X = X[0:data_size]
    if resolution is not None:
        X = np.array([resize(img, (resolution, resolution)) for img in X])
    X = transform(X)
    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)
    X = (X - minimum) / (maximum - minimum)
    X = (norm_max - norm_min) * X + norm_min
    X = np.expand_dims(X, 1)

    return X, Y


def get_dsprites_dataset(cache_dir='.', data_size=None, accelerator=None, norm_min=-1, norm_max=1):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if accelerator is None or accelerator.is_main_process:

        if not os.path.isfile(os.path.join(cache_dir, 'dsprites.npy')):
            url = 'https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true'
            download(url, os.path.isfile(os.path.join(cache_dir, 'dsprites.npy')))

    if accelerator is not None:
        accelerator.wait_for_everyone()

    d = np.load(os.path.join(cache_dir, 'dsprites.npy'))
    imgs = d['imgs']
    latents_values = d['latents_values']

    if data_size is not None:
        np.random.seed(0)
        idx = np.random.randint(0, high=len(imgs), size=data_size)
    else:
        idx = np.arange(len(imgs))
    X, Y = [], []
    for i in idx:
        X.append(imgs[i])
        Y.append([latents_values[i, 2], latents_values[i, 3]])
    X = np.array(X).astype(np.float32)
    Y = np.array(Y).astype(np.float32)

    minimum = np.min(Y, axis=0)
    maximum = np.max(Y, axis=0)
    Y = (Y - minimum) / (maximum - minimum)
    Y = np.expand_dims(Y, 1)

    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)
    X = (X - minimum) / (maximum - minimum)
    X = (norm_max - norm_min) * X + norm_min
    X = np.expand_dims(X, 1)

    return X, Y


def get_low_resolution(X, factor, channels_first=True):
    shape = X.shape
    if len(shape) == 4:
        X_down = [downscale_local_mean(img, (shape[1], factor, factor)) for img in X]
        if channels_first:
            X_up = [resize_local_mean(img, output_shape=(shape[2], shape[3]), channel_axis=0) for img in X_down]
        else:
            X_up = [resize_local_mean(img, output_shape=(shape[1], shape[2]), channel_axis=-1) for img in X_down]
    else:
        raise ValueError
    return np.array(X_up).astype(np.float32)
