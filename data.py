import os
import urllib

import numpy as np
from skimage.transform import resize, downscale_local_mean, resize_local_mean


def get_cmd_dataset(dataset_name, cache_dir='.', data_size=None, resolution=None, transform=np.log, accelerator=None):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if (accelerator is None or accelerator.is_main_process) and \
            not os.path.isfile(os.path.join(cache_dir, 'Maps_%s_LH_z=0.00.npy' % dataset_name)):
        urllib.request.urlretrieve(
            'https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/Maps_%s_LH_z=0.00.npy' % dataset_name,
            os.path.join(cache_dir, 'Maps_%s_LH_z=0.00.npy' % dataset_name)
        )

    if 'simba' in dataset_name.lower():
        parameter_file = 'params_SIMBA.txt'
    elif 'illustris' in dataset_name.lower():
        parameter_file = 'params_IllustrisTNG.txt'

    if (accelerator is None or accelerator.is_main_process) and \
            not os.path.isfile(os.path.join(cache_dir, parameter_file)):
        urllib.request.urlretrieve(
            'https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/%s' % parameter_file,
            os.path.join(cache_dir, parameter_file)
        )

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
    X = 2 * X - 1
    X = np.expand_dims(X, 1)

    return X, Y


def get_dsprites_dataset(cache_dir='.', data_size=None, accelerator=None):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if accelerator is None or accelerator.is_main_process:

        if not os.path.isfile(os.path.join(cache_dir, 'dsprites.npy')):
            urllib.request.urlretrieve(
                'https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true',
                os.path.join(cache_dir, 'dsprites.npy')
            )

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
    X = 2 * X - 1
    X = np.expand_dims(X, 1)

    return X, Y


def get_low_resolution(X, factor):
    shape = X.shape
    if len(shape) == 4:
        X_down = [downscale_local_mean(img, (shape[1], factor, factor)) for img in X]
        X_up = [resize_local_mean(img, output_shape=(shape[2], shape[3]), channel_axis=0) for img in X_down]
    return np.array(X_up).astype(np.float32)
