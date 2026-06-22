import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb
from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, test_3d_patch
from dataloaders.LADataset import LAHeart
from utils.utils import to_cuda, get_noise
from utils.BCP_utils import *
from utils.losses import *

from networks.Vnet import VNet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Datasets/la/data', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='CoDiff', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=6000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=12000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=8, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=1e-3, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--magnitude', type=float, default='10.0', help='magnitude')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
args = parser.parse_args()

unsup_weight_1 = 1.0
unsup_weight_2 = 0.5
mse_weight = 0.1

pre_max_iteration=args.pre_max_iteration
self_max_iteration=args.self_max_iteration
patch_size = (112, 112, 80)
num_classes = 2
c_batch_size = 2

def create_Vnet(ema=False):
    net = VNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_LA(masks)
    return masks


def LargestCC_LA(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()

train_data_path = args.root_path
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))


def get_XOR_region(mixout1, mixout2):
    s1 = torch.softmax(mixout1, dim=1)
    l1 = torch.argmax(s1, dim=1)

    s2 = torch.softmax(mixout2, dim=1)
    l2 = torch.argmax(s2, dim=1)

    diff_mask = (l1 != l2)
    return diff_mask

def pre_train(args, snapshot_path):
    model1 = create_Vnet()
    model2 = create_Vnet()

    trainset_lab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', reverse=True,
                             logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

    DICE = losses.mask_DiceLoss(nclass=2)

    model1.train()
    model2.train()
    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    iter_num = 0
    best_dice = 0
    best_dice2 = 0
    max_epoch = pre_max_iteration // len(lab_loader_a) + 1
    iterator = tqdm(range(1,max_epoch), ncols=70)
    for epoch_num in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            with torch.no_grad():
                _ , img_mask = context_mask(img_a, np.random.uniform(0.5, 0.7))
                img_mask = img_mask.unsqueeze(1)
            volume_batch_in = img_a * img_mask + img_b * (1 - img_mask)
            volume_batch_out = img_b * img_mask + img_a * (1 - img_mask)

            output_in_1, _ = model1(volume_batch_in)
            output_out_1, _ = model1(volume_batch_out)
            output_a_1=output_in_1*img_mask+output_out_1*(1 - img_mask)
            output_b_1=output_out_1*img_mask+output_in_1*(1 - img_mask)
            loss_ce = (F.cross_entropy(output_a_1, lab_a)+F.cross_entropy(output_b_1, lab_b))/2
            loss_dice = (DICE(output_a_1, lab_a)+DICE(output_b_1, lab_b))/2
            loss = (loss_ce + loss_dice) / 2

            output_in_2, _ = model2(volume_batch_in)
            output_out_2, _ = model2(volume_batch_out)
            output_a_2=output_in_2*img_mask+output_out_2*(1 - img_mask)
            output_b_2=output_out_2*img_mask+output_in_2*(1 - img_mask)
            loss_ce2 = (F.cross_entropy(output_a_2, lab_a)+F.cross_entropy(output_b_2, lab_b))/2
            loss_dice2 = (DICE(output_a_2, lab_a)+DICE(output_b_2, lab_b))/2
            loss2 = (loss_ce2 + loss_dice2) / 2

            iter_num += 1

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            logging.info(
                'iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' % (iter_num, loss, loss_dice, loss_ce))
            logging.info(
                'iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' % (iter_num, loss2, loss_dice2, loss_ce2))
            
        if iter_num >= 2000 and iter_num % 200 == 0:
            
            model1.eval()
            model2.eval()
            dice_sample = test_3d_patch.var_all_case_LA(model1, num_classes=num_classes, patch_size=patch_size,
                                                        stride_xy=18, stride_z=4)
            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}_1.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(snapshot_path, 'best_model_1.pth'.format(args.model))
                save_net_opt(model1, optimizer1, save_mode_path, epoch_num)
                save_net_opt(model1, optimizer1, save_best_path, epoch_num)
                logging.info("save best model1 to {}".format(save_mode_path))
            
            dice_sample2 = test_3d_patch.var_all_case_LA(model2, num_classes=num_classes, patch_size=patch_size,
                                                         stride_xy=18, stride_z=4)
            if dice_sample2 > best_dice2:
                best_dice2 = round(dice_sample2, 4)
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}_2.pth'.format(iter_num, best_dice2))
                save_best_path = os.path.join(snapshot_path, 'best_model_2.pth'.format(args.model))
                save_net_opt(model2, optimizer2, save_mode_path, epoch_num)
                save_net_opt(model2, optimizer2, save_best_path, epoch_num)
                logging.info("save best model2 to {}".format(save_mode_path))
            model1.train()
            model2.train()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    model1 = create_Vnet()
    model2 = create_Vnet()

    trainset_lab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', reverse=True,
                             logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_unlab', logging=logging)
    unlab_loader_a = DataLoader(trainset_unlab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_unlab', reverse=True,
                               logging=logging)
    unlab_loader_b = DataLoader(trainset_unlab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

    pretrained_model1 = os.path.join(pre_snapshot_path, 'best_model_1.pth')
    pretrained_model2 = os.path.join(pre_snapshot_path, 'best_model_2.pth')

    load_net_opt(model1, optimizer1, pretrained_model1)
    load_net_opt(model2, optimizer2, pretrained_model2)

    model1.train()
    model2.train()

    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    iter_num = 0
    best_dice = 0
    best_dice2 = 0
    mean_best_dice = 0
    max_epoch = self_max_iteration // len(lab_loader_a) + 1
    iterator = tqdm(range(1,max_epoch), ncols=70)
    for epoch in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(
                zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda(
                [img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])

            with torch.no_grad():
                img_1 = img_a + get_noise(img_a,'LA')
                img_2 = img_b + get_noise(img_a,'LA')
                unimg_1 = unimg_a + get_noise(img_a,'LA')
                unimg_2 = unimg_a + get_noise(img_a,'LA')
                
                output_p_1, feature_t_1 = model1(unimg_a)
                output_p_2, feature_t_2 = model2(unimg_a)

                plab_1 = get_cut_mask(output_p_1, nms=1).long()
                plab_2 = get_cut_mask(output_p_2, nms=1).long()

                mask_same = (plab_1 == plab_2).long()

                _, img_mask_1 = context_mask(img_a, np.random.uniform(0.5, 0.7))
                _, img_mask_2 = context_mask(img_b, np.random.uniform(0.5, 0.7))
                img_mask_1 = img_mask_1.unsqueeze(1)
                img_mask_2 = img_mask_2.unsqueeze(1)

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
            mse_u_1 = mask_mse_loss(outputs_unlab_1, plab_2, diff_mask=diff_mask_unlab*mask_f_2)
            mse_u_2 = mask_mse_loss(outputs_unlab_2, plab_1, diff_mask=diff_mask_unlab*mask_f_1)
            if loss_l_1 > loss_l_2:
                loss_1 = loss_l_1 + unsup_weight_1 * (loss_u_1 + mse_weight * mse_u_1)
                loss_2 = loss_l_2 + unsup_weight_2 * (loss_u_2 + mse_weight * mse_u_2)
            else:
                loss_1 = loss_l_1 + unsup_weight_2 * (loss_u_1 + mse_weight * mse_u_1)
                loss_2 = loss_l_2 + unsup_weight_1 * (loss_u_2 + mse_weight * mse_u_2)

            iter_num += 1

            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()
            print('epoch %d iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f, mse_u: %.4f \
               ' % (epoch, iter_num, loss_1, loss_l_1, loss_u_1, mse_u_1.item()))
            print('epoch %d iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f, mse_u: %.4f \
               ' % (epoch, iter_num, loss_2, loss_l_2, loss_u_2, mse_u_2.item()))

        if iter_num >= 2000 and iter_num % 200 == 0:
            model1.eval()
            model2.eval()
            dice_sample = test_3d_patch.var_all_case_LA(model1, num_classes=num_classes, patch_size=patch_size,
                                                        stride_xy=18, stride_z=4)
            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_dice_{}_1.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(self_snapshot_path, 'best_model_1.pth')
                torch.save(model1.state_dict(), save_mode_path)
                torch.save(model1.state_dict(), save_best_path)
                logging.info("save best model1 to {}".format(save_mode_path))
                logging.info("cur dice %.4f, max dice %.4f" % (dice_sample, best_dice))
            dice_sample2 = test_3d_patch.var_all_case_LA(model2, num_classes=num_classes, patch_size=patch_size,
                                                         stride_xy=18, stride_z=4)
            if dice_sample2 > best_dice2:
                best_dice2 = round(dice_sample2, 4)
                save_mode_path = os.path.join(self_snapshot_path,
                                              'iter_{}_dice_{}_2.pth'.format(iter_num, best_dice2))
                save_best_path = os.path.join(self_snapshot_path, 'best_model_2.pth')
                torch.save(model2.state_dict(), save_mode_path)
                torch.save(model2.state_dict(), save_best_path)
                logging.info("save best model2 to {}".format(save_mode_path))
                logging.info("cur dice %.4f, max dice %.4f" % (dice_sample2, best_dice2))
            mean_dice_sample = test_3d_patch.var_all_case_LA_mean(model1, model2, num_classes=num_classes,
                                                                  patch_size=patch_size, stride_xy=18, stride_z=4)
            if mean_dice_sample > mean_best_dice:
                mean_best_dice = round(mean_dice_sample, 4)
                save_mode_path1 = os.path.join(self_snapshot_path,
                                               'iter{}_{}_3.pth'.format(iter_num, mean_best_dice))
                save_best_path1 = os.path.join(self_snapshot_path, 'best_model_3.pth')

                save_mode_path2 = os.path.join(self_snapshot_path,
                                               'iter{}_{}_4.pth'.format(iter_num, mean_best_dice))
                save_best_path2 = os.path.join(self_snapshot_path, 'best_model_4.pth')

                torch.save(model1.state_dict(), save_mode_path1)
                torch.save(model1.state_dict(), save_best_path1)

                torch.save(model2.state_dict(), save_mode_path2)
                torch.save(model2.state_dict(), save_best_path2)

                logging.info("mean save best model to {}".format(save_mode_path1))
                logging.info("mean cur dice %.4f, max dice %.4f" % (mean_dice_sample, mean_best_dice))

            model1.train()
            model2.train()


if __name__ == "__main__":
    # make logger file
    pre_snapshot_path = "./model/CoDiff/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/CoDiff/LA_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('../code/train_LA.py', self_snapshot_path)
    # -- Pre-Training
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    self_train(args, pre_snapshot_path, self_snapshot_path)