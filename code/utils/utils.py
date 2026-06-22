import os
import sys
import shutil
import time
import random
import torch
import logging
from pathlib import Path

import numpy as np
from torch import multiprocessing
from torch.nn import functional as F
import nibabel as nib
from tensorboardX import SummaryWriter
from skimage.measure import label


def mkdir(path, level=2, create_self=True):
    """ Make directory for this path,
    level is how many parent folders should be created.
    create_self is whether create path(if it is a file, it should not be created)

    e.g. : mkdir('/home/parent1/parent2/folder', level=3, create_self=False),
    it will first create parent1, then parent2, then folder.

    :param path: string
    :param level: int
    :param create_self: True or False
    :return:
    """
    p = Path(path)
    if create_self:
        paths = [p]
    else:
        paths = []
    level -= 1
    while level != 0:
        p = p.parent
        paths.append(p)
        level -= 1

    for p in paths[::-1]:
        p.mkdir(exist_ok=True)


def seed_reproducer(seed=2022):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2020)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2020).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # set all gpus seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # if input data type and channels' changes arent' large use it improve train efficient
        torch.backends.cudnn.enabled = True


def cutmix_config_log(save_path, tensorboard=False):
    writer = SummaryWriter(str(save_path), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S')) if tensorboard else None

    save_path = str(Path(save_path) / 'log.txt')
    formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(message)s')

    logger = logging.getLogger(save_path.split('\\')[-2])
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(save_path)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(sh)

    return logger, writer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        return self


class Measures():
    def __init__(self, keys, writer, logger):
        self.keys = keys
        self.measures = {k: AverageMeter() for k in self.keys}
        self.writer = writer
        self.logger = logger

    def reset(self):
        [v.reset() for v in self.measures.values()]


def get_mask(out, thres=0.5):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).float()
    masks = masks[:, 1, :, :].contiguous()
    return masks


def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net(net, path):
    state = {
        'net': net.state_dict(),
    }
    torch.save(state, str(path))


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def generate_mask(img, patch_size):
    batch_l = img.shape[0]
    # batch_unlab = unimg.shape[0]
    loss_mask = torch.ones(batch_l, 96, 96, 96).cuda()
    # loss_mask_unlab = torch.ones(batch_unlab, 96, 96, 96).cuda()
    mask = torch.ones(96, 96, 96).cuda()
    w = np.random.randint(0, 96 - patch_size)
    h = np.random.randint(0, 96 - patch_size)
    z = np.random.randint(0, 96 - patch_size)
    mask[w:w + patch_size, h:h + patch_size, z:z + patch_size] = 0
    loss_mask[:, w:w + patch_size, h:h + patch_size, z:z + patch_size] = 0
    # loss_mask_unlab[:, w:w+patch_size, h:h+patch_size, z:z+patch_size] = 0
    # cordi = [w, h, z]
    return mask.long(), loss_mask.long()


def config_log(save_path, tensorboard=False):
    writer = SummaryWriter(str(save_path), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S')) if tensorboard else None

    save_path = str(Path(save_path) / 'log.txt')
    formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(message)s')

    logger = logging.getLogger(save_path.split('\\')[-2])
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(save_path)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(sh)

    return logger, writer


def to_cuda(tensors, device=None):
    res = []
    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            res.append(to_cuda(t, device))
        return res
    elif isinstance(tensors, (dict,)):
        res = {}
        for k, v in tensors.items():
            res[k] = to_cuda(v, device)
        return res
    else:
        if isinstance(tensors, torch.Tensor):
            if device is None:
                return tensors.cuda()
            else:
                return tensors.to(device)
        else:
            return tensors

def get_noise(img,dataset='LA'):
    if dataset=='LA':
        return torch.normal(mean=np.random.uniform(-0.5, 0.5), std=np.random.uniform(0, 0.5),
                     size=img.shape, device=img.device)
    elif dataset=='pancreas':
        return torch.normal(mean=np.random.uniform(-0.5, 0.5), std=np.random.uniform(0, 0),
                     size=img.shape, device=img.device)

def get_noise_acdc(img):
    return torch.normal(mean=np.random.uniform(0, 0.1), std=np.random.uniform(0, 0.05),
                 size=img.shape, device=img.device)

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)