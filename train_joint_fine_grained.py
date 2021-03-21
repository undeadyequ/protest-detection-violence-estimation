"""
created by: Donghyeon Won
Modified codes from
    http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    https://github.com/pytorch/examples/tree/master/imagenet
"""

from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import time
import shutil
#from itertools import ifilter
from PIL import Image
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, recall_score, precision_score

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models

from util import ProtestDataset, AverageMeter, Lighting, ProtestDataset_fts, ProtestDataset_txtfts
from easyocr.joint_model import JointVisDet, modified_resnet50, JointVisDetREC, vis_model


# for indexing output of the model
protest_demand_idx = Variable(torch.LongTensor([0]))
best_loss = float("inf")

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def evaluate(targets, predictions):
    performance = {
        'acc': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='macro'),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro')}
    return performance



def calculate_loss(output, target, criterions):
    """Calculate loss"""


    # number of protest images
    N_protest = int(target['protest_demand'].data.sum())

    demand_lb = target["protest_demand"].T[0].type(torch.int64)

    predictions = torch.argmax(output, dim=1)  # take argmax to get class id
    scores = {}

    predictions_cpu = predictions.cpu().detach().numpy()
    demand_lb_cpu = demand_lb.data.cpu().detach().numpy()
    #print(predictions_cpu)
    #print(demand_lb_cpu)

    scores['protest_demand_acc'] = [accuracy_score(predictions_cpu, demand_lb_cpu),
                                    precision_score(demand_lb_cpu, predictions_cpu, average='macro'),
                                    recall_score(demand_lb_cpu, predictions_cpu, average='macro'),
                                    f1_score(demand_lb_cpu, predictions_cpu, average='macro')]

    losses = criterions[0](output, demand_lb)
    return losses, scores, N_protest



def train(train_loader, model, criterions, optimizer, epoch):
    """training the model"""

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_protest = AverageMeter()
    loss_v = AverageMeter()
    protest_demand_acc = AverageMeter()
    sign_acc = AverageMeter()
    loss_protest = AverageMeter()

    end = time.time()
    loss_history = []
    for i, sample in enumerate(train_loader):
        # measure data loading batch_time
        input, target, text_enc = sample['image'], sample['label'], sample['text_fts']
        data_time.update(time.time() - end)

        if args.cuda:
            input = input.cuda()
            for k, v in target.items():
                target[k] = v.cuda()
            text_enc = text_enc.cuda()

        target_var = {}
        for k,v in target.items():
            target_var[k] = Variable(v)

        input_var = Variable(input)  # torch.Size([8, 3, 224, 224])
        output = model(input_var, text_enc)
        #output = model(input_var)


        losses, scores, N_protest = calculate_loss(output, target_var, criterions)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Evaluate
        protest_loss = losses.cpu().detach().numpy()

        loss_history.append(protest_loss)
        loss_protest.update(protest_loss, input.size(0))
        protest_demand_acc.update(scores['protest_demand_acc'][0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                  'Data {data_time.val:.2f} ({data_time.avg:.2f})  '
                  'Protest Loss {loss_val:.3f} ({loss_avg:.3f})  '
                  'Protest acc {protest_acc.val:.3f} ({protest_acc.avg:.3f})  '
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time,
                   loss_val=loss_protest.val,
                   loss_avg=loss_protest.avg,
                   protest_acc=protest_demand_acc))

    return loss_history

@torch.no_grad()
def validate(val_loader, model, criterions, epoch):
    """Validating"""
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_protest = AverageMeter()
    loss_v = AverageMeter()
    protest_demand_acc = AverageMeter()
    protest_demand_pre = AverageMeter()
    protest_demand_rec = AverageMeter()
    protest_demand_f1 = AverageMeter()

    end = time.time()
    loss_history = []
    for i, sample in enumerate(val_loader):
        # measure data loading batch_time
        input, target, text_enc = sample['image'], sample['label'], sample['text_fts']

        if args.cuda:
            input = input.cuda()
            for k, v in target.items():
                target[k] = v.cuda()
            text_enc = text_enc.cuda()

        target_var = {}
        for k,v in target.items():
            target_var[k] = Variable(v)

        input_var = Variable(input)  # torch.Size([8, 3, 224, 224])
        output = model(input_var, text_enc)
        #output = model(input_var)

        loss, scores, N_protest = calculate_loss(output, target_var, criterions)

        protest_loss = loss.cpu().detach().numpy()

        loss_history.append(protest_loss)
        loss_protest.update(protest_loss, input.size(0))
        protest_demand_acc.update(scores['protest_demand_acc'][0], input.size(0))
        protest_demand_pre.update(scores['protest_demand_acc'][1], input.size(0))
        protest_demand_rec.update(scores['protest_demand_acc'][2], input.size(0))
        protest_demand_f1.update(scores['protest_demand_acc'][3], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                  'Loss {loss_val:.3f} ({loss_avg:.3f})  '
                  'Protest Acc {protest_acc.val:.3f} ({protest_acc.avg:.3f})  '
                  .format(
                   epoch, i, len(val_loader), batch_time=batch_time,
                   loss_val =loss_protest.val,
                   loss_avg = loss_protest.avg,
                   protest_acc = protest_demand_acc))

    print(' * Loss {loss_avg:.3f} '
          'Acc {protest_acc.avg:.3f} '
          'Pre {protest_pre.avg:.3f} '
          'Rec {protest_rec.avg: 3f} '
          'f1 {protest_f1.avg: 3f}'
          .format(loss_avg=loss_protest.avg,
                  protest_acc=protest_demand_acc,
                  protest_pre=protest_demand_pre,
                  protest_rec=protest_demand_rec,
                  protest_f1=protest_demand_f1
                  ))
    return loss_protest.avg, loss_history


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.5 every 5 epochs"""
    lr = args.lr * (0.4 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoints"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def main():
    global best_loss
    loss_history_train = []
    loss_history_val = []
    data_dir = args.data_dir
    img_dir_train = os.path.join(data_dir, "train")
    img_dir_val = os.path.join(data_dir, "test")
    id_lab_trans_train = os.path.join(data_dir, "id_lab_trans_train.csv")
    id_lab_trans_eval= os.path.join(data_dir, "id_lab_trans_test.csv")
    id_lab_trans = os.path.join(data_dir, "id_lab_trans.csv")
    id_path_train = os.path.join(data_dir, "id_path_train.csv")
    id_path_eval = os.path.join(data_dir, "id_path_test.csv")

    # load pretrained resnet50 with a modified last fully connected layer
    #model = modified_resnet50()

    model = JointVisDetREC(tidim=1029, todim=50, vodim=10, odim=5)
    #model = vis_model(fc_out=5)

    # we need three different criterion for training
    criterion_protest_demand = nn.CrossEntropyLoss()
    criterions = [criterion_protest_demand]

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU Found")
    if args.cuda:
        model = model.cuda()
        criterions = [criterion.cuda() for criterion in criterions]
    # we are not training the frozen layers
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(
                        parameters, args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            loss_history_train = checkpoint['loss_history_train']
            loss_history_val = checkpoint['loss_history_val']
            if args.change_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    eigvec = torch.Tensor([[-0.5675,  0.7192,  0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948,  0.4203]])

    train_dataset = ProtestDataset_txtfts(
                        id_label_trans_train_f=id_lab_trans_train,
                        id_label_trans_f=id_lab_trans,
                        id_path_f=id_path_train,
                        transform=transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomRotation(30),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(
                                    brightness = 0.4,
                                    contrast = 0.4,
                                    saturation = 0.4,
                                    ),
                                transforms.ToTensor(),
                                Lighting(0.1, eigval, eigvec),
                                normalize,
                        ]))
    val_dataset = ProtestDataset_txtfts(
                    id_label_trans_train_f=id_lab_trans_eval,
                    id_label_trans_f=id_lab_trans,
                    id_path_f=id_path_eval,
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))
    train_loader = DataLoader(
                    train_dataset,
                    num_workers = args.workers,
                    batch_size = args.batch_size,
                    shuffle = True
                    )
    val_loader = DataLoader(
                    val_dataset,
                    num_workers = args.workers,
                    batch_size = args.batch_size)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        loss_history_train_this = train(train_loader, model, criterions,
                                        optimizer, epoch)
        loss_val, loss_history_val_this = validate(val_loader, model,
                                                   criterions, epoch)
        loss_history_train.append(loss_history_train_this)
        loss_history_val.append(loss_history_val_this)

        # loss = loss_val.avg

        is_best = loss_val < best_loss
        if is_best:
            print('best model!!')
        best_loss = min(loss_val, best_loss)


        save_checkpoint({
            'epoch' : epoch + 1,
            'state_dict' : model.state_dict(),
            'best_loss' : best_loss,
            'optimizer' : optimizer.state_dict(),
            'loss_history_train': loss_history_train,
            'loss_history_val': loss_history_val
        }, is_best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default = "UCLA-protest",
                        help = "directory path to UCLA-protest",
                        )
    parser.add_argument("--cuda",
                        action = "store_true",
                        help = "use cuda?",
                        )
    parser.add_argument("--workers",
                        type = int,
                        default = 4,
                        help = "number of workers",
                        )
    parser.add_argument("--batch_size",
                        type = int,
                        default = 8,
                        help = "batch size",
                        )
    parser.add_argument("--epochs",
                        type = int,
                        default = 100,
                        help = "number of epochs",
                        )
    parser.add_argument("--weight_decay",
                        type = float,
                        default = 1e-4,
                        help = "weight decay",
                        )
    parser.add_argument("--lr",
                        type = float,
                        default = 0.01,
                        help = "learning rate",
                        )
    parser.add_argument("--momentum",
                        type = float,
                        default = 0.9,
                        help = "momentum",
                        )
    parser.add_argument("--print_freq",
                        type = int,
                        default = 10,
                        help = "print frequency",
                        )
    parser.add_argument('--resume',
                        default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--change_lr',
                        action = "store_true",
                        help = "Use this if you want to \
                        change learning rate when resuming")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    args = parser.parse_args()

    if args.cuda:
        protest_idx = protest_demand_idx.cuda()
    main()
