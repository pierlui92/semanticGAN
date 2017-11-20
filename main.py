import argparse
import os
import scipy.misc
import numpy as np
import tensorflow as tf
from model import semanticgan

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='.', help='path of the dataset')
parser.add_argument('--step', dest='step', type=int, default=200000, help='# of step')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--load_size', dest='load_size', type=int, default=1024, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--dataset_dim',dest='dataset_dim',type=int,default=500,help='number of sample of dataset')
parser.add_argument('--ngf', dest='ngf', type=int, default=32, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA',choices=['AtoB','BtoA'])
parser.add_argument('--phase', dest='phase', help='train, test', required=True, choices=['train','test'])
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')

parser.add_argument('--test_dir', dest='test_dir', default='./testOut/', help='output folder of phase test')
parser.add_argument('--SoG', dest='SoG', default=True, action='store_false' ,help='if True semantic segmentation otherwise generation')

parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')

parser.add_argument('--sem_DA_fake_adversarial', dest='sem_DA_fake_adversarial', type=int, default=0, help='weight loss semantic discriminator fake adversarial')
parser.add_argument('--sem_DA_fake', dest='sem_DA_fake', type=int, default=1, help='weight loss semantic discriminator fake')
parser.add_argument('--sem_DA_real', dest='sem_DA_real', type=int, default=1, help='weight loss semantic discriminator real')
parser.add_argument('--sem_G_fake', dest='sem_G_fake', type=int, default=1, help='weight loss semantic generator fake')
parser.add_argument('--G', dest='G', type=int, default=1, help='weight loss GAN generator')
parser.add_argument('--DR', dest='DR', type=int, default=1, help='weight loss GAN discriminator real')
parser.add_argument('--DF', dest='DF', type=int, default=1, help='weight loss GAN discriminator fake')


parser.add_argument('--trainA', dest='trainA', default='./trainA', help='domain A train folder')
parser.add_argument('--trainASem', dest='trainASem', default='./trainASem', help='semantic map domain A train folder')
parser.add_argument('--trainB', dest='trainB', default='./trainB', help='domain B train folder')
parser.add_argument('--trainBSem', dest='trainBSem', default='./trainBSem', help='semantic map domain B train folder')
parser.add_argument('--testA', dest='testA', default='./testA', help='domain A test folder')
parser.add_argument('--testASem', dest='testASem', default='./testASem', help='semantic map domain A test folder')
parser.add_argument('--testB', dest='testB', default='./testB', help='domain B test folder')
parser.add_argument('--testBSem', dest='testBSem', default='./testBSem', help='semantic map domain B test folder')

parser.add_argument('--job-dir',help="this argument is ignored")


args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    print(type(args.SoG), args.SoG)
    with tf.Session() as sess:
        model = semanticgan(sess, args)
        if args.phase == 'train':
            model.train(args)
        else: 
            model.test(args)

if __name__ == '__main__':
    tf.app.run()