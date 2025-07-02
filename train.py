import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.seg_modeling import PIPO_Model as Seg_net
from networks.seg_modeling import CONFIGS as CONFIGS_Seg
from trainer import trainer_MyoPS
import timeit

start = timeit.default_timer()


parser = argparse.ArgumentParser(description="基于傅里叶特征解耦与Prompt学习的多模态心肌病理分割")
parser.add_argument('--root_path', type=str,
                    default='/data/fangdg/MyoPS++/process/cine_sa/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='QS_Seg_MyoPS', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='/home/fangdg/P2/PIPO/list', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=128, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--name', type=str,
                    default='R50', help='model name')
parser.add_argument("--mode_type", type=str, default="random", help="mode type, random, 0, 1, 2, 3, 4, 5, 6")
parser.add_argument("--eval_mode", type=str, default="all", help="mode type, random, 0, 1, 2, 3, 4, 5, 6")
parser.add_argument("--exp_name", type=str, default="baseline")
parser.add_argument("--train_only", action="store_true",default=False, help="train only, no evaluation")


args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'QS_Seg_MyoPS': {
            'root_path': '/data/fangdg/MyoPS++/process/cine_sa/train_npz',
            'root_path1':'/data/fangdg/MyoPS++/process/psir/train_npz',
            'root_path2':'/data/fangdg/MyoPS++/process/t2w/train_npz',
            'list_dir': '/home/fangdg/P2/PIPO_net/list',
            'num_classes': 4,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.root_path1 = dataset_config[dataset_name]['root_path1']
    args.root_path2 = dataset_config[dataset_name]['root_path2']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.exp = dataset_name
    snapshot_path = "/home/fangdg/P2/PIPO_net/results/{}".format(args.exp)
    snapshot_path += '_' + args.name + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    snapshot_path = snapshot_path + '_' + args.exp_name
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config = CONFIGS_Seg[args.name]
    config.n_classes = args.num_classes
    config.n_skip = args.n_skip
    net = Seg_net(config, img_size=args.img_size, num_classes=config.n_classes).cuda()

    trainer_MyoPS(args, net, snapshot_path)

end = timeit.default_timer()
print(f"Training completed in {end - start:.2f} seconds.")
