#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Author: cc
  @CreateTime:   2017-12-18T20:20:50+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""

from __future__ import print_function

import argparse
import random
import chainer
from chainer import Variable
import numpy as np
import pandas as pd
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.serializers import npz
from utils import load_dataset, img_reader


class Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self, nb_class=1000):
        super(Alex, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  96, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256,  5, pad=2)
            self.conv3 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv4 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv5 = L.Convolution2D(None, 256,  3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, nb_class)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        return h


class Classifier(chainer.Chain):
    """
    classifier for source and target dataset
    inputs:
        sharedNet: alexnet model
        alpha: weight for coral loss
        mode: with/without coral loss
    """

    def __init__(self, sharedNet, alpha=0.1):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.sharedNet = sharedNet
            self.alpha = alpha

    def __call__(self, src_x, src_y, tar_x, tar_y):
        features_s = self.sharedNet(src_x)
        features_t = self.sharedNet(tar_x)
        cls_loss_s = F.softmax_cross_entropy(features_s, src_y)
        cls_loss_t = F.softmax_cross_entropy(features_t, tar_y)
        xp = self.sharedNet.xp
        coral_loss = coral_func(xp, features_s, features_t)
        acc_s = F.accuracy(features_s, src_y)
        acc_t = F.accuracy(features_t, tar_y)
        loss = cls_loss_s + self.alpha * coral_loss
        chainer.report({'loss': loss, 'cls_loss_s': cls_loss_s, 'cls_loss_t': cls_loss_t,
                        'coral_loss': self.alpha * coral_loss, 'src_acc': acc_s, 'tar_acc': acc_t}, self)
        return loss


def coral_func(xp, src, tar):
    """
    inputs:
        -src(Variable) : features extracted from source data
        -tar(Variable) : features extracted from target data
    return coral loss between source and target features
    ref: Deep CORAL: Correlation Alignment for Deep Domain Adaptation \
         (https://arxiv.org/abs/1607.01719
    """
    ns, nt = src.data.shape[0], tar.data.shape[0]
    dim = src.data.shape[1]

    ones_s = xp.ones((1, ns), dtype=np.float32)
    ones_t = xp.ones((1, nt), dtype=np.float32)
    tmp_s = F.matmul(Variable(ones_s), src)
    tmp_t = F.matmul(Variable(ones_t), tar)
    cov_s = (F.matmul(F.transpose(src), src) -
             F.matmul(F.transpose(tmp_s), tmp_s) / ns) / (ns - 1)
    cov_t = (F.matmul(F.transpose(tar), tar) -
             F.matmul(F.transpose(tmp_t), tmp_t) / nt) / (nt - 1)

    coral = F.sum(F.squared_error(cov_s, cov_t)) / (4 * dim * dim)
    return coral


def main():
    parser = argparse.ArgumentParser(description='Chainer-DeepCORAL')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--alpha', '-a', type=float, default=1.0,
                        help='Weight for coral loss')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epochs))
    print('')
    alexnet = Alex()
    
    nb_cls = 31

    # load pretrain alexnet weights
    npz.load_npz("alexnet.npz", alexnet)

    # change fc8 layer to output 31 class
    initializer = chainer.initializers.Normal(0.005)
    alexnet.fc8 = L.Linear(None, nb_cls, initialW=initializer)
    model = Classifier(alexnet, args.alpha)
    if chainer.cuda.available:
        model.to_gpu()
    # initial setting of original paper
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9
    # Setup an optimizer
    optimizer = chainer.optimizers.MomentumSGD(
        lr=LEARNING_RATE, momentum=MOMENTUM)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))
    # change learning rate of fc8 layer
    model.sharedNet.fc8.W.update_rule.hyperparam.lr = 10 * LEARNING_RATE
    model.sharedNet.fc8.b.update_rule.hyperparam.lr = 10 * LEARNING_RATE

    source_train = load_dataset("dataset/office31/amazon/train")
    source_val = load_dataset("dataset/office31/amazon/val")
    target_train = load_dataset("dataset/office31/webcam/train")
    target_val = load_dataset("dataset/office31/webcam/val")

    # shuffle data of source and target every epoch
    src_train_idx = list(range(0, source_train.shape[0]))
    random.shuffle(src_train_idx)
    source_train = source_train.iloc[src_train_idx]
    tar_train_idx = list(range(0, target_train.shape[0]))
    random.shuffle(tar_train_idx)
    target_train = target_train.iloc[tar_train_idx * 3]

    nb_train = min(source_train.shape[0], target_train.shape[0])
    src_train_x, src_train_y = img_reader(source_train[:nb_train])
    tar_train_x, tar_train_y = img_reader(target_train[:nb_train])

    nb_val = min(source_val.shape[0], target_val.shape[0])
    src_val_x, src_val_y = img_reader(source_val[:nb_val])
    tar_val_x, tar_val_y = img_reader(target_val[:nb_val])

    print("{} samples for training, {} samples for validation.".format(
        nb_train, nb_val))
    train_data = chainer.datasets.TupleDataset(
        src_train_x, src_train_y, tar_train_x, tar_train_y)
    val_data = chainer.datasets.TupleDataset(
        src_val_x, src_val_y, tar_val_x, tar_val_y)

    train_iter = chainer.iterators.SerialIterator(
        train_data, args.batchsize, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(
        val_data, args.batchsize, repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(
        log_name='Log_with_{}_coral.json'.format(args.alpha)))

    # save loss and accuracy curve
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/coral_loss', 'main/cls_loss_s', 'validation/main/coral_loss', 'validation/main/cls_loss_s'],
                'epoch', file_name='Loss_curve_with_{}_coral.png'.format(args.alpha)))
        trainer.extend(
            extensions.PlotReport(
                ['main/src_acc', 'validation/main/src_acc', 'main/tar_acc', 'validation/main/tar_acc'],
                'epoch', file_name='Accuracy_curve_with_{}_coral.png'.format(args.alph))

    # print logs
    trainer.extend(extensions.PrintReport(
        ['epoch',
         'main/cls_loss_s', 'main/coral_loss',
         'main/src_acc', 'main/tar_acc',
         'validation/main/src_acc', 'validation/main/tar_acc']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
