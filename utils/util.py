import importlib
import os

import cv2 as cv
import numpy as np
import torch
from natsort import natsorted
from torchvision import transforms
from torchvision.utils import save_image

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def read_cube_to_np(img_dir, stack_axis=2, cvflag=cv.IMREAD_GRAYSCALE):
    assert os.path.exists(img_dir), f"got {img_dir}"
    print(img_dir)
    imgs = []
    names = natsorted(os.listdir(img_dir))
    for name in names:
        img = cv.imread(os.path.join(img_dir, name), cvflag)
        imgs.append(img)
    imgs = np.stack(imgs, axis=stack_axis)
    return imgs


def read_cube_to_tensor(path, stack_axis=1, cvflag=cv.IMREAD_GRAYSCALE):
    imgs = []
    names = natsorted(os.listdir(path))
    for name in names:
        img = cv.imread(os.path.join(path, name), cvflag)
        img = transforms.ToTensor()(img)
        imgs.append(img)
    imgs = torch.stack(imgs, dim=stack_axis)
    return imgs


def save_cube_from_tensor(img, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    for j in range(img.shape[0]):
        img_path = os.path.join(result_dir, str(j + 1) + '.png')
        save_image(img[j, :, :], img_path)


def save_cube_from_numpy(data, result_name, tonpy=False):
    if tonpy:
        np.save(result_name + '.npy', data)
    else:
        result_dir = result_name
        if not os.path.exists(result_dir): os.makedirs(result_dir)
        for i in range(data.shape[0]):
            cv.imwrite(os.path.join(result_dir, str(i + 1) + '.png'), data[i, ...])


def get_file_path_from_dir(src_dir, file_name):
    for root, dirs, files in os.walk(src_dir):
        if file_name in files:
            print(root, dirs, files)
            return os.path.join(root, file_name)