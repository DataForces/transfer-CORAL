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


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(args.unit, 10))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    return 0


def coral_func(src, tar):
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
    xp = model.xp
    ones_s = xp.ones((1, ns), dtype=np.float32)
    ones_t = xp.ones((1, nt), dtype=np.float32)
    tmp_s = F.matmul(Variable(ones_s), src)
    tmp_t = F.matmul(Variable(ones_t), tar)
    cov_s = (F.matmul(F.transpose(src), src) -
             F.matmul(F.transpose(tmp_s), tmp_s) / ns) / (ns-1)
    cov_t = (F.matmul(F.transpose(tar), tar) -
             F.matmul(F.transpose(tmp_t), tmp_t) / nt) / (nt-1)

    coral = F.sum(F.squared_error(cov_s, cov_t))/(4*dim*dim)
    return coral


def batch_train():

    # building model for training
    model = Alex()
    used_pretrain = False
    nb_cls = 31
    if used_pretrain:
        # load pretrain weights
        npz.load_npz("alexnet.npz", model)
        # replace final fc layer

    initializer = chainer.initializers.Normal(0.005)
    model.fc8 = L.Linear(None, nb_cls, initialW=initializer)

    if chainer.cuda.available:
        model.to_gpu()
    # change fc8 layer to output 31 class
    xp = model.xp
    batch_sizes = [32, 32]

    source_train = load_dataset("dataset/office31/amazon/train")
    source_val = load_dataset("dataset/office31/amazon/val")
    target_train = load_dataset("dataset/office31/webcam/train")
    target_val = load_dataset("dataset/office31/webcam/val")

#    train_steps = min(source_train.shape[0]//batch_sizes[0],
#                      target_train.shape[0]//batch_sizes[1])
    train_steps = 1
    alpha = 0.0
    nb_epochs = 50
    has_coral = False
    iter_nb = 0
    for epoch in range(nb_epochs):
        cls_s, cls_t, coral_st = 0, 0, 0
        acc_s, acc_t = 0, 0
        # shuffle data of source and target every epoch
        src_train_idx = list(range(0, source_train.shape[0]))
        random.shuffle(src_train_idx)
        source_train = source_train.iloc[src_train_idx]
        tar_train_idx = list(range(0, target_train.shape[0]))
        random.shuffle(tar_train_idx)
        target_train = target_train.iloc[tar_train_idx]
        for i in range(train_steps):
            iter_src_df = source_train.iloc[i * batch_sizes[0]:(i+1) * batch_sizes[0]]
            iter_tar_df = target_train.iloc[i * batch_sizes[1]:(i+1) * batch_sizes[1]]
            iter_src_x, iter_src_y = img_reader(iter_src_df)
            iter_tar_x, iter_tar_y = img_reader(iter_tar_df)
            source_data = Variable(xp.array(iter_src_x))
            source_label = Variable(xp.array(iter_src_y))
            target_data = Variable(xp.array(iter_tar_x))
            target_label = Variable(xp.array(iter_tar_y))
            fc8_t = model(target_data)
            fc8_s = model(source_data)
            model.cleargrads()
            cls_loss_s = F.softmax_cross_entropy(fc8_s, source_label)
            cls_loss_t = F.softmax_cross_entropy(fc8_t, target_label)
            cls_acc_s = F.accuracy(fc8_s, source_label)
            cls_acc_t = F.accuracy(fc8_t, target_label)
            if has_coral:
                coral_loss = coral_func(fc8_s, fc8_t)
                loss = cls_loss_s + coral_loss*alpha
            else:
                loss = cls_loss_s
            # Setup an optimizer
            optimizer = chainer.optimizers.Adam()
            optimizer.setup(model)

            loss.backward()
            optimizer.update()
            cls_s += cls_loss_s.data
            cls_t += cls_loss_t.data
            if has_coral:
                coral_st += coral_loss.data
            else:
                coral_st += 0
            acc_s += cls_acc_s.data
            acc_t += cls_acc_t.data
            iter_nb += 1
        cls_s /= train_steps
        cls_t /= train_steps
        coral_st /= train_steps
        acc_s /= train_steps
        acc_t /= train_steps
        print("Epoch:{:02d}, Iter:{:03d}, src-Loss:{:0.2f}, tar-Loss:{:0.2f}, src-Acc:{:0.2f}, tar-Acc:{:0.2f}".
              format(epoch, iter_nb, float(cls_s), float(cls_t), float(acc_s), float(acc_t)))


if __name__ == '__main__':
    gpu = -1
    batch_size = 32
    epochs = 20
    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batch_size))
    print('# epoch: {}'.format(20))
    print('')

    alexnet = Alex()
    used_pretrain = False
    nb_cls = 31
    if used_pretrain:
        # load pretrain weights
        npz.load_npz("alexnet.npz", alexnet)

    # change fc8 layer to output 31 class
    initializer = chainer.initializers.Normal(0.005)
    alexnet.fc8 = L.Linear(None, nb_cls, initialW=initializer)
    model = L.Classifier(alexnet)
    if chainer.cuda.available:
        model.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    source_train = load_dataset("dataset/office31/amazon/train")
    source_val = load_dataset("dataset/office31/amazon/val")
    target_train = load_dataset("dataset/office31/webcam/train")
    target_val = load_dataset("dataset/office31/webcam/val")
    src_train_x, src_train_y = img_reader(source_train)
    src_val_x, src_val_y = img_reader(source_val)
    train_data = chainer.datasets.TupleDataset(src_train_x, src_train_y)
    val_data = chainer.datasets.TupleDataset(src_val_x, src_val_y)

    train_iter = chainer.iterators.SerialIterator(train_data, batch_size, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(val_data, batch_size, repeat=False, shuffle=False)
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out='result')

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()
