from __future__ import absolute_import, division, print_function

import os
import argparse

import torch

from envs import TicTacToeEnv
from model import ES
from train import train_loop, render_env

parser = argparse.ArgumentParser(description='ES')
parser.add_argument('--lr', type=float, default=0.3, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-decay', type=float, default=1, metavar='LRD',
                    help='learning rate decay')
parser.add_argument('--sigma', type=float, default=0.05, metavar='SD',
                    help='noise standard deviation')
parser.add_argument('--wd', type=float, default=0.996, metavar='WD',
                    help='amount of weight decay')
parser.add_argument('--n', type=int, default=40, metavar='N',
                    help='batch size, must be even')
parser.add_argument('--restore', default='', metavar='RES',
                    help='checkpoint from which to restore')
parser.add_argument('--variable-ep-len', action='store_true',
                    help="Change max episode length during training")
parser.add_argument('--silent', action='store_true',
                    help='Silence print statements during training')
parser.add_argument('--test', action='store_true',
                    help='Just render the env, no training')
parser.add_argument('--max-gradient-updates', type=int, default=100000,
                    metavar='MGU', help='maximum number of updates')


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.n % 2 == 0

    chkpt_dir = 'checkpoints/'
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    env = TicTacToeEnv()
    synced_model = ES(env.observation_space, env.action_space)
    for param in synced_model.parameters():
        param.requires_grad = False
    if args.restore:
        state_dict = torch.load(args.restore)
        synced_model.load_state_dict(state_dict)

    if args.test:
        render_env(synced_model)
    else:
        train_loop(args, synced_model, chkpt_dir)
