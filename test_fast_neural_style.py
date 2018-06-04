from __future__ import print_function
import os
import pickle
import argparse
import cv2
import numpy as np
from PIL import Image

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers

from chainer.links import caffe as chainer_caffe

from model import ImageTransformer
from updater import StyleUpdater, display_image
from dataset import SuperImageDataset

str2list = lambda x: x.split(';')
str2bool = lambda x:x.lower() == 'true'

def make_optimizer(model, alpha):
    optimizer = chainer.optimizers.Adam(alpha=alpha)
    optimizer.setup(model)
    return optimizer

def original_colors(original, stylized):
    y, _, _ = cv2.split(cv2.cvtColor(stylized, cv2.COLOR_BGR2YUV))
    _, u, v = cv2.split(cv2.cvtColor(original, cv2.COLOR_BGR2YUV))
    yuv_img = cv2.merge((y, u, v))
    bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    return bgr_img

def main():
    parser = argparse.ArgumentParser(description='Fast neural style transfer')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--filter_num', type=int, default=32, help="# of filters in ImageTransformer's 1st conv layer")
    parser.add_argument('--output_channel', type=int, default=3, help='# of output image channels')
    parser.add_argument('--tanh_constant', type=float, default=150, help='Constant for output of ImageTransformer')
    parser.add_argument('--instance_normalization', type=str2bool, default=False, help='Use InstanceNormalization if True')
    parser.add_argument('--model_path', default='fast_style_result/transformer_iter.npz', help='Path for pretrained model')
    parser.add_argument('--out', default='fast_style_result', help='Directory to output the result')

    args = parser.parse_args()

    print('Input arguments:')
    for key, value in vars(args).items():
        print('\t{}: {}'.format(key, value))
    print('')

    G = ImageTransformer(args.filter_num, args.output_channel, args.tanh_constant, args.instance_normalization)
    serializers.load_npz(args.model_path, G)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        D.to_gpu()
    print('Load models done ...\n')

    # orig_img = cv2.imread("./fast_style_result/my.png", cv2.IMREAD_COLOR)[:, :, ::-1]
    orig_img = cv2.imread("./fast_style_result/my.png", cv2.IMREAD_COLOR)
    img = cv2.resize(orig_img, (256, 256))
    h, w, _ = img.shape
    img_a = np.asarray(img, dtype=np.float32)
    img_a = np.transpose(img_a, (2, 0, 1))
    A = chainer.cuda.to_cpu(G(np.asarray([img_a])).data)
    # import pdb; pdb.set_trace()
    A = np.asarray(np.transpose(A[0], [1, 2, 0]) + np.array([103.939, 116.779, 123.68]), dtype=np.uint8)
    # A = original_colors(Image.fromarray(img), Image.fromarray(A))
    A = original_colors(orig_img, A)
    # cv2.imshow('image', np.asarray(A, dtype=np.uint8))
    cv2.imshow('image', A)
    cv2.waitKey(0)
    # Image.fromarray(A).


if __name__ == '__main__':
    with chainer.using_config('train', False):
        main()
