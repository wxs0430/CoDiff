import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import (ACDCDataSet, BaseDataSets, RandomGenerator, TwoStreamBatchSampler,
                                 ThreeStreamBatchSampler)
from networks.unet import UNet, UNet_2d
from utils import losses, val_2d
from utils.utils import to_cuda, get_noise_acdc
from test_ACDC import TESTACDC

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Datasets/acdc', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='CoDiff', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=15000, help='maximum epoch number to train')
parser.add_argument('--self_max_iteration', type=int, default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=1e-3, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--magnitude', type=float, default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int, default=6, help='multinum of random masks')

args = parser.parse_args()
DICE = losses.DiceLoss(n_classes=4)
CE = nn.CrossEntropyLoss(reduction='none')
pre_max_iteration=args.pre_max_iteration
self_max_iteration=args.self_max_iteration
unsup_weight_1 = 1.0
unsup_weight_2 = 0.5
mse_weight = 0.05
c_batch_size = 12


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()


def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)
    return probs

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * np.random.uniform(0.5, 0.7)), int(img_y * np.random.uniform(0.5, 0.7))
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def mask_loss(output, img_l, mask=None):
    img_l = img_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    if mask == None:
        dice_loss = DICE(output_soft, img_l.unsqueeze(1))
        loss_ce =  F.cross_entropy(output, img_l)
    else:
        dice_loss = DICE(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1))
        loss_ce = (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss = (loss_ce+dice_loss)/2
    return loss


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax - target_logits) ** 2
    return mse_loss


def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def mask_mse_loss(net3_output, img_l, diff_mask=None):
    img_l_onehot = to_one_hot(img_l.unsqueeze(1), 4)
    mse_loss = torch.mean(softmax_mse_loss(net3_output, img_l_onehot), dim=1)

    loss = torch.sum(diff_mask * mse_loss) / (torch.sum(diff_mask) + 1e-16)
    return loss


def get_XOR_region(mixout1, mixout2):
    s1 = torch.softmax(mixout1, dim=1)
    l1 = torch.argmax(s1, dim=1)
    s2 = torch.softmax(mixout2, dim=1)
    l2 = torch.argmax(s2, dim=1)

    diff_mask = (l1 != l2)
    return diff_mask


def pre_train(args, snapshot_path):
    torch.use_deterministic_algorithms(False)
    num_classes = args.num_classes
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model1 = UNet_2d(in_chns=1, class_num=num_classes).cuda()
    model2 = UNet_2d(in_chns=1, class_num=num_classes).cuda()

    db_val = ACDCDataSet(base_dir=args.root_path, split="val", logging=logging)

    trainset_lab_a = ACDCDataSet(base_dir=args.root_path, split="train_lab",
                                 transform=transforms.Compose([RandomGenerator(args.patch_size)]), logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = ACDCDataSet(base_dir=args.root_path, split="train_lab",
                                 transform=transforms.Compose([RandomGenerator(args.patch_size)]), reverse=True,
                                 logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer = optim.Adam(model1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
    logging.info("optim.Adam pre_training")

    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainset_lab_a)))

    model1.train()
    model2.train()

    iter_num = 0
    best_performance = 0.0
    best_performance2 = 0.0
    max_epoch = pre_max_iteration // len(lab_loader_a) + 1
    iterator = tqdm(range(1,max_epoch), ncols=70)
    for epoch in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            with torch.no_grad():
                img_mask, _ = generate_mask(img_a)
                lab_a = lab_a.type(torch.int64)
                lab_b = lab_b.type(torch.int64)

            # -- original
            volume_batch_in = img_a * img_mask + img_b * (1 - img_mask)
            volume_batch_out = img_b * img_mask + img_a * (1 - img_mask)

            output_in_1, _ = model1(volume_batch_in)
            output_out_1, _ = model1(volume_batch_out)
            output_a_1 = output_in_1 * img_mask + output_out_1 * (1 - img_mask)
            output_b_1 = output_out_1 * img_mask + output_in_1 * (1 - img_mask)

            loss =(mask_loss(output_a_1,lab_a)+mask_loss(output_b_1,lab_b))/2

            output_in_2, _ = model2(volume_batch_in)
            output_out_2, _ = model2(volume_batch_out)
            output_a_2 = output_in_2 * img_mask + output_out_2 * (1 - img_mask)
            output_b_2 = output_out_2 * img_mask + output_in_2 * (1 - img_mask)
            loss_2 = (mask_loss(output_a_2,lab_a)+mask_loss(output_b_2,lab_b))/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()

            iter_num += 1

            logging.info('iteration %d: loss: %f' % (iter_num, loss))

        if iter_num >= 2000 and iter_num % 200 == 0:
            model1.eval()
            model2.eval()
            metric_list = 0.0
            metric_list_2 = 0.0
            for _, (img_val, lab_val) in tqdm(enumerate(valloader), ncols=70):
                metric_i = val_2d.test_single_volume(img_val, lab_val, model1, classes=num_classes)
                metric_i_2 = val_2d.test_single_volume(img_val, lab_val, model2, classes=num_classes)

                metric_list += np.array(metric_i)
                metric_list_2 += np.array(metric_i_2)

            metric_list = metric_list / len(db_val)
            metric_list_2 = metric_list_2 / len(db_val)

            performance = np.mean(metric_list, axis=0)[0]
            performance2 = np.mean(metric_list_2, axis=0)[0]

            if performance > best_performance:
                best_performance = performance
                save_mode_path = os.path.join(snapshot_path,
                                              'epoch_{}_dice_{}_1.pth'.format(epoch, round(best_performance, 4)))
                save_best_path = os.path.join(snapshot_path, 'best_model_1.pth')
                save_net_opt(model1, optimizer, save_mode_path)
                save_net_opt(model1, optimizer, save_best_path)

            if performance2 > best_performance2:
                best_performance2 = performance2
                save_mode_path = os.path.join(snapshot_path,
                                              'epoch_{}_dice_{}_2.pth'.format(epoch, round(best_performance2, 4)))
                save_best_path = os.path.join(snapshot_path, 'best_model_2.pth')
                save_net_opt(model2, optimizer2, save_mode_path)
                save_net_opt(model2, optimizer2, save_best_path)

            logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
            logging.info(
                'resnet iteration %d : mean_dice : %f' % (iter_num, performance2))
            model1.train()
            model2.train()


def self_train(args, pre_snapshot_path, snapshot_path):
    torch.use_deterministic_algorithms(False)
    num_classes = args.num_classes
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    pre_trained_model1 = os.path.join(pre_snapshot_path, 'best_model_1.pth')
    pre_trained_model2 = os.path.join(pre_snapshot_path, 'best_model_2.pth')

    model1 = UNet_2d(in_chns=1, class_num=num_classes).cuda()
    model2 = UNet_2d(in_chns=1, class_num=num_classes).cuda()

    db_val = ACDCDataSet(base_dir=args.root_path, split="val", logging=logging)

    trainset_lab_a = ACDCDataSet(base_dir=args.root_path, split="train_lab",
                                 transform=transforms.Compose([RandomGenerator(args.patch_size)]), logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = ACDCDataSet(base_dir=args.root_path, split="train_lab",
                                 transform=transforms.Compose([RandomGenerator(args.patch_size)]), reverse=True,
                                 logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_a = ACDCDataSet(base_dir=args.root_path, split="train_unlab",
                                   transform=transforms.Compose([RandomGenerator(args.patch_size)]), logging=logging)
    unlab_loader_a = DataLoader(trainset_unlab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_b = ACDCDataSet(base_dir=args.root_path, split="train_unlab",
                                   transform=transforms.Compose([RandomGenerator(args.patch_size)]), reverse=True,
                                   logging=logging)
    unlab_loader_b = DataLoader(trainset_unlab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

    load_net_opt(model1, optimizer1, pre_trained_model1)

    load_net_opt(model2, optimizer2, pre_trained_model2)

    logging.info("Start self_training")

    model1.train()
    model2.train()

    iter_num = 0
    best_performance = 0.0
    best_performance2 = 0.0
    best_performance_mean = 0.0
    max_epoch = self_max_iteration // len(lab_loader_a) + 1
    iterator = tqdm(range(1,max_epoch), ncols=70)
    for epoch in iterator:
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(
                zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda(
                [img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])
            with torch.no_grad():
                lab_a = lab_a.type(torch.int64)
                lab_b = lab_b.type(torch.int64)
                img_1 = img_a + get_noise_acdc(img_a)
                img_2 = img_b + get_noise_acdc(img_a)
                unimg_1 = unimg_a + get_noise_acdc(img_a)
                unimg_2 = unimg_a + get_noise_acdc(img_a)

                output_p_1, feature_t_1 = model1(unimg_a)
                output_p_2, feature_t_2 = model2(unimg_a)

                plab_1 = get_ACDC_masks(output_p_1, nms=1).long()
                plab_2 = get_ACDC_masks(output_p_2, nms=1).long()

                mask_same = (plab_1 == plab_2).long()

                img_mask_1, _ = generate_mask(img_a)
                img_mask_2, _ = generate_mask(img_a)

            mix_in_1 = unimg_1 * img_mask_1 + img_1 * (1 - img_mask_1)
            mix_out_1 = img_1 * img_mask_1 + unimg_1 * (1 - img_mask_1)
            outputs_in_1, feature_in_1 = model1(mix_in_1)
            outputs_out_1, feature_out_1 = model1(mix_out_1)
            outputs_lab_1 = outputs_in_1 * (1 - img_mask_1) + outputs_out_1 * img_mask_1
            outputs_unlab_1 = outputs_in_1 * img_mask_1 + outputs_out_1 * (1 - img_mask_1)
            feature_1 = feature_in_1 * img_mask_1 + feature_out_1 * (1 - img_mask_1)

            mix_in_2 = unimg_2 * img_mask_2 + img_2 * (1 - img_mask_2)
            mix_out_2 = img_2 * img_mask_2 + unimg_2 * (1 - img_mask_2)
            outputs_in_2, feature_in_2 = model2(mix_in_2)
            outputs_out_2, feature_out_2 = model2(mix_out_2)
            outputs_lab_2 = outputs_in_2 * (1 - img_mask_2) + outputs_out_2 * img_mask_2
            outputs_unlab_2 = outputs_in_2 * img_mask_2 + outputs_out_2 * (1 - img_mask_2)
            feature_2 = feature_in_2 * img_mask_2 + feature_out_2 * (1 - img_mask_2)

            with torch.no_grad():
                diff_mask_unlab = get_XOR_region(outputs_unlab_1, outputs_unlab_2).long()
                mask_f_1 = F.cosine_similarity(feature_t_1, feature_1)
                mask_f_2 = F.cosine_similarity(feature_t_2, feature_2)

            loss_l_1 = mask_loss(outputs_lab_1, lab_a)
            loss_l_2 = mask_loss(outputs_lab_2, lab_b)
            loss_u_1 = mask_loss(outputs_unlab_1, plab_2, mask_same)
            loss_u_2 = mask_loss(outputs_unlab_2, plab_1, mask_same)
            mse_u_1 = mask_mse_loss(outputs_unlab_1, plab_2, diff_mask=diff_mask_unlab * mask_f_2)
            mse_u_2 = mask_mse_loss(outputs_unlab_2, plab_1, diff_mask=diff_mask_unlab * mask_f_1)
            if loss_l_1 > loss_l_2:
                loss_1 = (loss_l_1) + unsup_weight_1 * (loss_u_1 + mse_weight * mse_u_1)
                loss_2 = (loss_l_2) + unsup_weight_2 * (loss_u_2 + mse_weight * mse_u_2)
            else:
                loss_1 = (loss_l_1) + unsup_weight_2 * (loss_u_1 + mse_weight * mse_u_1)
                loss_2 = (loss_l_2) + unsup_weight_1 * (loss_u_2 + mse_weight * mse_u_2)

            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()

            iter_num += 1

            print('epoch %d iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f, mse_u: %.4f \
               ' % (epoch, iter_num, loss_1, loss_l_1, loss_u_1, mse_u_1.item()))
            print('epoch %d iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f, mse_u: %.4f \
               ' % (epoch, iter_num, loss_2, loss_l_2, loss_u_2, mse_u_2.item()))

            if iter_num >= 2000 and iter_num % 200 == 0:
                model1.eval()
                model2.eval()
                metric_list = 0.0
                metric_list_2 = 0.0
                metric_list_mean = 0.0

                for _, (img_val, lab_val) in tqdm(enumerate(valloader), ncols=70):
                    metric_i = val_2d.test_single_volume(img_val, lab_val, model1, classes=num_classes)
                    metric_i_2 = val_2d.test_single_volume(img_val, lab_val, model2, classes=num_classes)
                    metric_i_mean = val_2d.test_single_volume_mean(img_val, lab_val, model1, model2, classes=num_classes)

                    metric_list += np.array(metric_i)
                    metric_list_2 += np.array(metric_i_2)
                    metric_list_mean += np.array(metric_i_mean)

                metric_list = metric_list / len(db_val)
                metric_list_2 = metric_list_2 / len(db_val)
                metric_list_mean = metric_list_mean / len(db_val)

                performance = np.mean(metric_list, axis=0)[0]
                performance2 = np.mean(metric_list_2, axis=0)[0]
                performance_mean = np.mean(metric_list_mean, axis=0)[0]

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}_1.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, 'best_model_1.pth')
                    save_net_opt(model1, optimizer1, save_mode_path)
                    save_net_opt(model1, optimizer1, save_best_path)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}_2.pth'.format(iter_num,
                                                                                   round(best_performance2, 4)))
                    save_best_path = os.path.join(snapshot_path, 'best_model_2.pth')
                    save_net_opt(model2, optimizer2, save_mode_path)
                    save_net_opt(model2, optimizer2, save_best_path)

                if performance_mean > best_performance_mean:
                    best_performance_mean = performance_mean

                    save_mode_path1 = os.path.join(snapshot_path, 'iter_{}_dice_{}_3.pth'.format(iter_num, round(
                        best_performance_mean, 4)))
                    save_best_path1 = os.path.join(snapshot_path, 'best_model_3.pth')

                    save_mode_path2 = os.path.join(snapshot_path, 'iter_{}_dice_{}_4.pth'.format(iter_num, round(
                        best_performance_mean, 4)))
                    save_best_path2 = os.path.join(snapshot_path, 'best_model_4.pth')

                    save_net_opt(model1, optimizer1, save_mode_path1)
                    save_net_opt(model1, optimizer1, save_best_path1)

                    save_net_opt(model2, optimizer2, save_mode_path2)
                    save_net_opt(model2, optimizer2, save_best_path2)

                TESTACDC(iter_num, phase='self_train')
                logging.info(
                    'iteration %d : mean_dice : %f' % (iter_num, performance))
                logging.info(
                    'resnet iteration %d : mean_dice : %f' % (
                    iter_num, performance2))
                logging.info('mean iteration %d : mean_dice : %f' % (
                    iter_num, performance_mean))

                model1.train()
                model2.train()


if __name__ == "__main__":
    # -- path to save models
    pre_snapshot_path = "./model/CoDiff/ACDC_{}_{}_labeled/pre_train".format(args.exp, 7)
    self_snapshot_path = "./model/CoDiff/ACDC_{}_{}_labeled/self_train".format(args.exp, 7)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('../code/train_ACDC.py', self_snapshot_path)

    # Pre_train
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        cudnn.benchmark = False
        # cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    pre_train(args, pre_snapshot_path)

    # Self_train
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    self_train(args, pre_snapshot_path, self_snapshot_path)