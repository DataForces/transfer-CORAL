#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
  @Author: cc
  @CreateTime:   2017-12-21T21:02:24+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import glob
import os
import random
import shutil
import numpy as np
import pandas as pd
import cv2
from chainer import Variable



def separate_train_val(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, 'train')):
        os.mkdir(os.path.join(args.output_dir, 'train'))

    if not os.path.exists(os.path.join(args.output_dir, 'val')):
        os.mkdir(os.path.join(args.output_dir, 'val'))

    directories = os.listdir(args.root)

    for dir_index, dir_name in enumerate(directories):
        files = glob.glob(os.path.join(args.root, dir_name, '*.jpg'))
        random.shuffle(files)
        # skip empty dir
        if len(files) == 0:
            continue
        edge = int(len(files)*args.ratio)
        # copy orginal data to training set
        for file_index, file_path in enumerate(files[:edge]):
                target_dir = os.path.join(args.output_dir, 'train', dir_name)
                if not os.path.exists(target_dir):
                    os.mkdir(target_dir)
                shutil.copy(file_path, target_dir)
                print('Copied {} => {}'.format(file_path, target_dir))

        # copy orginal data to validation set
        for file_index, file_path in enumerate(files[edge:]):
                target_dir = os.path.join(args.output_dir, 'val', dir_name)
                if not os.path.exists(target_dir):
                    os.mkdir(target_dir)
                shutil.copy(file_path, target_dir)
                print('Copied {} => {}'.format(file_path, target_dir))



def img_reader(batch_df,
               img_row=227,
               img_col=227,
               crop = None,
               subtract="imagenet",
               direction="chw",
               mode="bgr"):
    """
    inputs:
        batch_df(pandas - DataFrame): batch record of img_paths and labels
    returns:
        batch_x(np.array), batch_y(np.array)
    """
    x, y = [], []
    for _, record in batch_df.iterrows():
        img = cv2.imread(record["paths"])

        if crop == "random":
            h, w = img.shape
            # Randomly crop a region of the image
            top = random.randint(0, h - img_row - 1)
            left = random.randint(0, w - img_col - 1)
            img = img[:, top:top + img_row, left:left + img_col]
        elif crop == "center":
            h, w = img.shape
            # Crop the center
            top = (h - img_row) // 2
            left = (w - img_col) // 2
            img = img[:, top:top + img_row, left:left + img_col]
        else:
            img = cv2.resize(img, (img_col, img_row))
        img = (img / 255).astype("float32")
        if subtract == "imagenet":
            # means of imagenet brg[0.407, 0.458, 0.485]
            means = [0.407, 0.458, 0.485]
            img[:, :, 0] -= means[0]
            img[:, :, 1] -= means[1]
            img[:, :, 2] -= means[2]
        if mode == "rgb":
            img = img[:, :, ::-1]
        if direction == 'chw':
            img = img.transpose((2, 0, 1))
        x.append(np.expand_dims(img, axis=0))
        y.append(record["labels"])
    return np.concatenate(x), y


def generate_labels(dataset_dir):
    # generate label for training and validation images
    dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    dirs.sort()
    img_paths, labels = [], []
    label_dict = []
    for _label, _dir in enumerate(dirs):
        label_dict.append([_dir, _label])
        files = glob.glob(os.path.join(dataset_dir, _dir, '*.jpg'))
        img_paths.append(files)
        labels.append([_label]*len(files))

    img_paths, labels = np.concatenate(img_paths), np.concatenate(labels)
    records = pd.DataFrame()
    records["paths"] = img_paths
    records["labels"] = labels
    records.to_csv(dataset_dir+"-records.csv", index=False)
    return label_dict


def load_dataset(datadir):
    """
    inputs:
        datadir(str): dir of dataset, such as 'dataset/office31/amazon/train'
    return records(pandas-DataFrame):
        [paths](str): path of image files
        [labels](int): labels of image
    """
    records = pd.read_csv(datadir+"-records.csv")
    print("[Info] Loading data from {}".format(datadir))
    return records


if __name__ == '__main__':
    # # seperate training and validation datasets
    # for datadir in ["amazon", "dslr", "webcam"]:
    #     parser = argparse.ArgumentParser(description='converter')
    #     parser.add_argument('--root', default='dataset/office31/{}/ori'.format(datadir))
    #     parser.add_argument('--output_dir', default='dataset/office31/{}'.format(datadir))
    #     parser.add_argument('--ratio', type=float, default=0.7)
    #     parser.add_argument('--resize', type=list, default=[256, 256])
    #     args = parser.parse_args()
    #
    #     separate_train_val(args)

    # # save records images in training and validation datasets
    # label_dicts=[]
    # for datadir in ["amazon", "dslr", "webcam"]:
    #     for sets in ["ori", "train", "val"]:
    #         dataset_dir = 'dataset/office31/{}/{}'.format(datadir, sets)
    #         label_dicts.append(generate_labels(dataset_dir))
    # label_dicts = [np.array(i) for i in label_dicts]
    # for _dict in label_dicts[1:]:
    #     assert np.sum(label_dicts[0]==_dict) == label_dicts[0].shape[0]*label_dicts[0].shape[1]
    # df = load_dataset('dataset/office31/amazon/train')
    #
    # batch_size = 32
    # train_steps = 10
    # epochs = 1
    # for epoch in range(epochs):
    #     rec_idx = random.shuffle(list(range(0, df.shape[0])))
    #     for i in range(train_steps):
    #         iter_df = df.iloc[i*batch_size:(i+1)*batch_size]
    #         iter_x, iter_y = img_reader(iter_df)
