import torch
from utils.module import *
import argparse


LEARNING_RATE = 0.05
BATCH_SIZE = 100

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # check cuda

# parse our input parameters
parse = argparse.ArgumentParser()
parse.add_argument('--size', '-s', help='Sample size', default=3000, type=int)
parse.add_argument('--epoch', '-e', help='Numbers of epoch', default=20, type=int)
parse.add_argument('--val', '-v', help='Size of validation sample', default=0.1, type=float)
parse.add_argument('--test', '-t', help='Size of test sample', default=0.2, type=float)
args = parse.parse_args()

args = parse.parse_args()
NUM_EPOCHS = args.epoch
samples_size = args.size
test_size = args.test
val_size = args.val

# start train and check model
train(NUM_EPOCHS=NUM_EPOCHS, samples_size=samples_size, test_size=test_size, val_size=val_size, DEVICE=DEVICE)
