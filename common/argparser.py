import argparse

from common import util


parser = argparse.ArgumentParser()

parser.add_argument('-note', type=str, default='debug')

# environment setting
parser.add_argument('-n_servers', type=int, default=10)
parser.add_argument('-n_resources', type=int, default=2)
parser.add_argument('-n_tasks', type=int, default=None,
                    help='Use all tasks by default.')
parser.add_argument('-w1', type=float, default=1e-4)
parser.add_argument('-w2', type=float, default=1e-4)
parser.add_argument('-w3', type=float, default=1e-4)
parser.add_argument('-P_0', type=int, default=87)
parser.add_argument('-P_100', type=int, default=145)
parser.add_argument('-T_on', type=int, default=30)
parser.add_argument('-T_off', type=int, default=30)

# global-tier setting
parser.add_argument('-g_lr', type=float, default=1e-5)
parser.add_argument('-g_ac', type=str, default='sigmoid')
parser.add_argument('-g_opt', type=str, default='adam')
parser.add_argument('-g_latents', type=str, default='128,64')

# local-tier setting
parser.add_argument('-l_lr', type=float, default=1e-5)
parser.add_argument('-l_ac', type=str, default='sigmoid')
parser.add_argument('-l_opt', type=str, default='adam')
parser.add_argument('-l_latents', type=str, default='128,64')
parser.add_argument('-l_obsize', type=int, default=100)
parser.add_argument('-l_actsize', type=int, default=1)

# dqn setting
parser.add_argument('-replace_target_iter', type=int, default=int(1e2))
parser.add_argument('-memory_size', type=int, default=64)

# experiment setting
parser.add_argument('-gpu', type=str, default='-1')
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-batch_size', type=int, default=5)
parser.add_argument('-total_epoches', type=int, default=int(1e5))

# a2c setting
parser.add_argument('-gamma', type=float, default=0.8)

args = parser.parse_args()

args.g_latents = util.split_integers(args.g_latents)
args.l_latents = util.split_integers(args.l_latents)


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
