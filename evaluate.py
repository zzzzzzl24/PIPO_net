import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

from datasets.dataset_MyoPS import MyoPS_dataset
from utils import test_single_volume
from networks.seg_modeling import PIPO_Model as Seg_net
from networks.seg_modeling import CONFIGS as CONFIGS_Seg

mode_name = ['bSSFP', 'LGE', 'T2w', 'bSSFP-LGE', 'bSSFP-T2w', 'LGE-T2w', 'bSSFP-LGE-T2w']

def inference(args, model, test_save_path=None):
    if args.eval_mode == "all":
        class_2_dice = 0
        class_2_hd95 = 0
        class_4_dice = 0
        class_4_hd95 = 0
        for n in range(0, 7):
            args.eval_mode = n
            db_test = args.Dataset(base_dir=args.volume_path, base_dir1=args.volume_path1, base_dir2=args.volume_path2, split="test_vol", list_dir=args.list_dir, mode_type=args.eval_mode)
            testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
            logging.info('{}'.format(mode_name[n]))
            metric_list = 0.0
            for i_batch, sampled_batch in tqdm(enumerate(testloader)):
                # h, w = sampled_batch["image"].size()[2:]
                image, iamge1, image2, label, mode, case_name = sampled_batch["image"], sampled_batch["image1"], sampled_batch["image2"], sampled_batch["label"], sampled_batch["mode"], sampled_batch['case_name'][0]
                metric_i = test_single_volume(image, iamge1, image2, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                        test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing, mode_Type=mode)
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
        logging.info(f"Average Dice for Class 2: {class_2_dice / 7:.4f}, Average Dice for Class 4: {class_4_dice / 7:.4f}")
        dice_score = (class_2_dice / 7  + class_4_dice / 7) /2
        args.eval_mode = "all"
        return dice_score 

    else:
        db_test = args.Dataset(base_dir=args.volume_path, base_dir1=args.volume_path1, base_dir2=args.volume_path2, split="test_vol", list_dir=args.list_dir, mode_type=args.eval_mode)
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
        # model.eval()
        metric_list = 0.0
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, iamge1, image2, label, mode, case_name = sampled_batch["image"], sampled_batch["image1"], sampled_batch["image2"], sampled_batch["label"], sampled_batch["mode"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, iamge1, image2, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                        test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing, mode_Type=mode)
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(db_test)
        dice_score = np.mean(metric_list[[1, 3], 0])
        for i in range(1, args.num_classes + 2):
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        return dice_score 

def evaluate(args_dict,current_epochs, exp_name):
    args = args_dict
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'QS_Seg_MyoPS': {
            'Dataset': MyoPS_dataset,
            'volume_path': './MyoPS++/process/cine_sa/test_vol_h5',
            'volume_path1': './MyoPS++/process/psir/test_vol_h5',
            'volume_path2': './MyoPS++/process/t2w/test_vol_h5',
            'list_dir': './P2/PIPO_net/list',
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

    experiment_name = f"{exp_name}"
    snapshot_path = f"./P2/PIPO_net/results/{dataset_name}_{args.name}_{args.img_size}_skip{args.n_skip}"
    if args.max_epochs!= 30:
        snapshot_path += f'_epo{args.max_epochs}'
    snapshot_path += f'_bs{args.batch_size}'
    if args.base_lr != 0.01:
        snapshot_path += f'_lr{args.base_lr}'
    if args.seed != 1234:
        snapshot_path += f'_s{args.seed}'
    snapshot_path += f'_{experiment_name}'

    config = CONFIGS_Seg[args.name]
    config.n_classes = args.num_classes
    config.n_skip = args.n_skip
    net = Seg_net(config, img_size=args.img_size, num_classes=config.n_classes).cuda()

    snapshot_file = snapshot_path + f"/epoch_{current_epochs}.pth"
    checkpoint = torch.load(snapshot_file)
    new_state_dict = OrderedDict()
    if args.n_gpu > 1:
        for k, v in checkpoint.items():
            name = k[7:]  # remove "module." prefix
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(checkpoint)

    return inference(args, net)
