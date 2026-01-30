import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_MyoPS import MyoPS_dataset
from utils import test_single_volume, filter_txt_file, extract_scar_edema_metrics_to_csv
from networks.seg_modeling import PIPO_Model as Seg_net
from networks.seg_modeling import CONFIGS as CONFIGS_Seg
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default="", help='fold id: 0..4')
parser.add_argument('--run_id', type=int, default="", help='repeat id within a fold: 0..4')
parser.add_argument('--volume_path', type=str,
                    default='', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='QS_Seg_MyoPS', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='', help='list dir')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=128, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", default="True", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--name', type=str, default='R50', help='model name')

parser.add_argument('--test_save_dir', type=str, default='/home/fangdg/P2/PIPO_net/predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--mode_type", type=str, default="6", help="mode type, 0, 1, 2, 3, 4, 5, 6")
parser.add_argument("--only_one", action="store_true", default=False, help="only one mode inference, default is False")
args = parser.parse_args()


mode_name = ['bSSFP', 'LGE', 'T2w', 'bSSFP-LGE', 'bSSFP-T2w', 'LGE-T2w', 'bSSFP-LGE-T2w']
#单一实验
def inference_one(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, base_dir1=args.volume_path1, base_dir2=args.volume_path2, split="test_vol", list_dir=args.list_dir, mode_type=args.mode_type)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    logging.info('{}'.format(mode_name[int(args.mode_type)]))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, iamge1, image2, label, mode, case_name = sampled_batch["image"], sampled_batch["image1"], sampled_batch["image2"], sampled_batch["label"], sampled_batch["mode"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, iamge1, image2, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing, mode_Type=mode)
        for idx, (dice, hd95) in enumerate(metric_i):
            logging.info(f"Class {idx + 1} - Dice: {dice:.4f}, HD95: {hd95:.4f}")
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes+2):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    return "Testing Finished!"

# 全部实验
def inference(args, model, test_save_path=None):
    model.eval()
    class_2_dice = 0
    class_2_hd95 = 0
    class_4_dice = 0
    class_4_hd95 = 0
    for n in range(0, 7):
        args.mode_type = n
        db_test = args.Dataset(base_dir=args.volume_path, base_dir1=args.volume_path1, base_dir2=args.volume_path2, split="test_vol", list_dir=args.list_dir, mode_type=args.mode_type)
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
        logging.info("{} test iterations per epoch".format(len(testloader)))
        logging.info('{}'.format(mode_name[n]))
        metric_list = 0.0
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            # h, w = sampled_batch["image"].size()[2:]
            image, iamge1, image2, label, mode, case_name = sampled_batch["image"], sampled_batch["image1"], sampled_batch["image2"], sampled_batch["label"], sampled_batch["mode"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, iamge1, image2, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing, mode_Type=mode)
            for idx, (dice, hd95) in enumerate(metric_i):
                logging.info(f"Class {idx + 1} - Dice: {dice:.4f}, HD95: {hd95:.4f}")
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(db_test)
        for i in range(1, args.num_classes+2):
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
            if i == 2:
                class_2_dice += metric_list[i-1][0]
                class_2_hd95 += metric_list[i-1][1]
            if i == 4:
                class_4_dice += metric_list[i-1][0]
                class_4_hd95 += metric_list[i-1][1]
    logging.info(f"Average Dice for Class 2: {class_2_dice / 7:.4f}, Average HD95 for Class 2: {class_2_hd95 / 7:.4f}")
    logging.info(f"Average Dice for Class 4: {class_4_dice / 7:.4f}, Average HD95 for Class 4: {class_4_hd95 / 7:.4f}")

    return "Testing Finished!"

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
    if args.exp_name == "" or args.exp_name is None:
        args.exp_name = f"response_fold_{args.fold}_run_{args.run_id}"
    dataset_config = {
        'QS_Seg_MyoPS': {
            'Dataset': MyoPS_dataset,
            'volume_path': f'/data/fangdg/P2/MyoPS380/process/fold_{args.fold}/cine_sa/test_vol_h5',
            'volume_path1': f'/data/fangdg/P2/MyoPS380/process/fold_{args.fold}/psir/test_vol_h5',
            'volume_path2': f'/data/fangdg/P2/MyoPS380/process/fold_{args.fold}/t2w/test_vol_h5',
            'list_dir': f'/data/fangdg/P2/MyoPS380/process/fold_{args.fold}/list',
            'num_classes': 4,
            'z_spacing': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.volume_path1 = dataset_config[dataset_name]['volume_path1']
    args.volume_path2 = dataset_config[dataset_name]['volume_path2']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = dataset_name
    exp_name = args.exp_name  # You can change this to your desired experiment name
    snapshot_path = "/home/fangdg/P2/PIPO_net/results/{}".format(args.exp)
    snapshot_path += '_' + args.name + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    snapshot_path = snapshot_path + '_' + f"{exp_name}"

    config = CONFIGS_Seg[args.name]
    config.n_classes = args.num_classes
    config.n_skip = args.n_skip
    net = Seg_net(config, img_size=args.img_size, num_classes=config.n_classes).cuda()

    snapshot = "{}/epoch_299.pth".format(snapshot_path)
    checkpoint = torch.load(snapshot)
    new_state_dict = OrderedDict()
    if args.n_gpu > 1:
        for k, v in checkpoint.items():
            name = k[7:]  # 去掉 "module." 前缀
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(checkpoint)
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = '/home/fangdg/P2/PIPO_net/test_log/' + args.exp
    folder = log_folder.replace('test_log', 'test_log1')
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)


    if args.is_savenii:
        args.test_save_dir = '/home/fangdg/P2/PIPO_net/predictions/{}'.format(exp_name)
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
        
    else:
        test_save_path = None
    if args.only_one:
        inference_one(args, net, test_save_path)
    else:
        inference(args, net, test_save_path)

    filter_txt_file(log_folder + '/'+snapshot_name+".txt", folder + '/'+snapshot_name+".txt")
    extract_scar_edema_metrics_to_csv(folder + '/'+snapshot_name+".txt", folder + '/'+snapshot_name+".csv")
