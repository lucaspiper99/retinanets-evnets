import os
import math
import torch
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T, datasets
import pandas as pd


class TinyImageNETDataset(Dataset):
 
    def __init__(
        self,
        path: str,
        train: bool,
        preload: bool,
        transform: callable = None,
    ):

        self.data = None
        self.train = train
        self.preload = preload
        self.transform = transform
        
        folder = datasets.ImageFolder(os.path.join(path, 'train'))
        if self.train:
            self.paths = [f[0] for f in folder.imgs]
            self.labels = torch.tensor([f[1] for f in folder.imgs])
        else:
            val_data = pd.read_csv(os.path.join(path, "val", 'val_annotations.txt'), 
                       sep='\t', 
                       header=None, 
                       usecols=[i for i in range(2)],
                       names=['Path', 'Class']
                       )
            self.paths = val_data['Path'].apply(
                lambda x: os.path.join(path, "val", 'images', x)
                ).to_list()
            self.labels = torch.tensor([folder.class_to_idx[c] for c in val_data['Class']]) 

        if not self.transform:
            self.transform = T.Compose([
                T.ConvertImageDtype(dtype=torch.float),
                T.Resize((64, 64)),
                T.Lambda(lambda img: img.expand(3, -1, -1)),
                T.Normalize(mean=[.5]*3, std=[.5]*3),
            ])

        if self.preload:
            self.data = []
            with tqdm(total=len(self.paths), desc=f"Preloading dataset") as pbar:
                for path in self.paths:
                    self.data.append(self._load_image(path))
                    pbar.update(1)
 
    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, idx):
        x = None
        y = self.labels[idx]
        if self.preload:
            x = self.transform(self.data[idx])
        else:
            x = self.transform(self._load_image(self.paths[idx]))
        return x, y
    
    def _load_image(self, path):
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = T.ToTensor()(img)
            return img

    def get_X(self):
        return self.transform(torch.stack(self.data)).numpy()
    
    def get_y(self):
        return self.labels.numpy()


class TinyImageNET_CDataset(Dataset):
 
    def __init__(
        self,
        path: str,
        corruption: str,
        severity: int,
        preload: bool,
        transform: callable = None,
    ):

        self.data = None
        self.preload = preload
        self.transform = transform
        
        folder = datasets.ImageFolder(os.path.join(path, corruption, str(severity)))
        self.paths = [f[0] for f in folder.imgs]
        self.labels = torch.tensor([f[1] for f in folder.imgs])

        if not self.transform:
            self.transform = T.Compose([
                T.ConvertImageDtype(dtype=torch.float),
                T.Resize((64, 64), antialias=True),
                T.Lambda(lambda img: img.expand(3, -1, -1)),
                T.Normalize(mean=[.5]*3, std=[.5]*3),
            ])

        if self.preload:
            self.data = []
            with tqdm.tqdm(total=len(self.paths), desc=f"Preloading dataset") as pbar:
                for path in self.paths:
                    self.data.append(self._load_image(path))
                    pbar.update(1)
 
    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, idx):
        x = None
        y = self.labels[idx]
        if self.preload:
            x = self.transform(self.data[idx])
        else:
            x = self.transform(self._load_image(self.paths[idx]))
        return x, y

    def _load_image(self, path):
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = T.ToTensor()(img)
            return img


def gaussian_kernel(
    sigma: float, k: float=1, size:float=15, norm:bool=False
    ) -> torch.tensor:
    """Returns a 2D Gaussian kernel.

    :param sigma (float): standard deviation of the Gaussian
    :param k (float, optional): height of the Gaussian
    :param size (float, optional): kernel size
    :param norm (bool, optional): whether no normalize the kernel
    :return: gaussian kernel
    """
    assert size % 2 == 1
    w = size // 2
    grid_val = torch.arange(-w, w+1, dtype=torch.float)
    x, y = torch.meshgrid(grid_val, grid_val, indexing='ij')
    gaussian = k * torch.exp(-(x**2 + y**2) / (2*(sigma)**2))
    if norm: gaussian /= torch.abs(gaussian.sum())
    return gaussian


def dog_kernel(
    sigma_c: float, sigma_s: float, k_c: float, k_s: float,
    polarity:int, size:int=21
    ) -> torch.tensor:
    """Returns a 2D Difference-of-Gaussians kernel.

    :param sigma_c: standard deviation of the center Gaussian
    :param sigma_s: standard deviation of the surround Gaussian
    :param k_c: peak sensitivity of the center
    :param k_s: peak sensitivity of the surround
    :param polarity: polarity of the center Gaussian (+1 or -1)
    :param size: kernel size
    :return: difference-of-gaussians kernel
    """
    assert size % 2 == 1
    assert polarity in [-1 , 1]
    center_gaussian = gaussian_kernel(sigma=sigma_c, k=k_c, size=size)
    surround_gaussian = gaussian_kernel(sigma=sigma_s, k=k_s, size=size)
    dog = polarity * (center_gaussian - surround_gaussian)
    dog /= torch.sum(dog)
    return dog

def circular_kernel(size:int, radius:float) -> torch.tensor:
    """Returns circular kernel.

    :param size (int): kernel size
    :param radius (float): radius of the circle
    :return: circular kernel
    """

    w = size // 2
    grid_val = torch.arange(-w, w+1, dtype=torch.float)
    x, y = torch.meshgrid(grid_val, grid_val, indexing='ij')
    kernel = torch.zeros(y.shape)
    kernel[torch.sqrt(x**2 + y**2) <= radius] = 1
    kernel /= torch.sum(kernel)
    return kernel

def gabor_kernel(
    frequency:float,  sigma_x:float, sigma_y:float,
    theta:float=0, offset:float=0, ks:int=61
    ):
    """Returns gabor kernel.

    :param frequency (float): spatial frequency of gabor
    :param sigma_x (float): standard deviation in x direction
    :param sigma_y (float): standard deviation in y direction
    :param theta (int, optional): Angle theta. Defaults to 0.
    :param offset (int, optional): Offset. Defaults to 0.
    :param ks (int, optional): Kernel size. Defaults to 61.
    :return: np.ndarray: 2-dimensional Gabor kernel
    """
    w = ks // 2
    grid_val = torch.arange(-w, w+1, dtype=torch.float)
    x, y = torch.meshgrid(grid_val, grid_val)
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    g = torch.zeros(y.shape)
    g[:] = torch.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= torch.cos(2 * np.pi * frequency * rotx + offset)
    return g

def generate_grating(
    size:int, radius:float, sf:float, theta:float=0, phase:float=0,
    contrast:float=1, gaussian_mask:bool=False
    ) -> torch.tensor:
    """Returns masked grating array.

    :param size (int): kernel size
    :param radius (float): standard deviation times sqrt(2) of the mask if gaussian_mask is True, and the radius if is false
    :param sf (float): spatial frequency of the grating
    :param theta (float, optional): angle of the grating 
    :param phase (float, optional): phase of the grating
    :param gaussian_mask (bool, optional): mask is a Gaussian if true and a circle if false 
    :param contrast (float, optional): maximum contrast of the grating
    :return: 2d masked grating array
    """
    grid_val = torch.linspace(-size//2, size//2+1, size, dtype=torch.float)
    X, Y = torch.meshgrid(grid_val, grid_val, indexing='ij')
    grating = torch.sin(2*math.pi*sf*(X*math.cos(theta) + Y*math.sin(theta)) + phase) * contrast
    mask = torch.exp(-((X**2 + Y**2)/(2*(radius/np.sqrt(2))**2))) if gaussian_mask else torch.sqrt(X**2 + Y**2) <= radius
    return grating * mask * .5 + .5


def sample_dist(hist:np.array, bins:int, ns:float, scale:str='linear'):
    """_summary_

    Args:
        hist (np.array): histogram
        bins (int): number of bins
        ns (float): sample size
        scale (str, optional): distributino scale. Defaults to 'linear'.

    :returns rand_sample (np.array): 
    """    
    rand_sample = np.random.rand(ns)
    if scale == 'linear':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), bins)
    elif scale == 'log2':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), np.log2(bins))
        rand_sample = 2**rand_sample
    elif scale == 'log10':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), np.log10(bins))
        rand_sample = 10**rand_sample
    return rand_sample

def set_seed(seed):
    """Enforces deterministic behaviour and sets RNG seed for numpy and pytorch.

    :param seed (int): seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
