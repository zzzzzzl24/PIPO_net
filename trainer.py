import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, SoftmaxWeightedLoss, DomainClsLoss, Compute_multi_modal_loss

from torchvision import transforms
from evaluate import evaluate

def trainer_MyoPS(args, model, snapshot_path):
    from datasets.dataset_MyoPS import MyoPS_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    distribution_loss = nn.L1Loss()
    loss_domain_cls = DomainClsLoss()
    # max_iterations = args.max_iterations
    db_train = MyoPS_dataset(base_dir=args.root_path, base_dir1=args.root_path1, base_dir2=args.root_path2, list_dir=args.list_dir, split="train", mode_type=args.mode_type,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    # ce_loss = SoftmaxWeightedLoss(num_classes)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    best_model_num = 0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, image1_batch, image2_batch, label_batch, mode_Type = sampled_batch['image'], sampled_batch['image1'], sampled_batch['image2'], sampled_batch['label'], sampled_batch['mode']
            image_batch, image1_batch, image2_batch, label_batch, mode_Type = image_batch.cuda(), image1_batch.cuda(), image2_batch.cuda(), label_batch.cuda(), mode_Type.cuda()
            out_pre, share_f, spec_logits, mode_Type = model(image_batch, image1_batch, image2_batch, mode_Type = mode_Type)
            
            out_cross_loss = ce_loss(out_pre, label_batch)
            out_dice_loss = dice_loss(out_pre, label_batch, softmax=True)
            loss = 0.5* out_cross_loss + 0.5* out_dice_loss
            
            dis_shared_loss, dis_spec_loss = Compute_multi_modal_loss(
                                                share_f, spec_logits, mode_Type,
                                                distribution_loss, loss_domain_cls
                                             )

            loss = loss + 0.1*dis_shared_loss + 0.02*dis_spec_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', out_dice_loss, iter_num)

            logging.info('iteration %d : loss : %f, loss_dice: %f' % (iter_num, loss.item(), out_dice_loss.item()))
            # print('iteration %d : loss : %f, loss_fuse: %f' % (iter_num, loss.item(), out_dice_loss.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                out_pre = torch.argmax(torch.softmax(out_pre, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', out_pre[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        
        save_interval = 20  # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 5:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            

        # 保存模型
        if not args.train_only and ((epoch_num + 1) % save_interval == 0 or epoch_num >= max_epoch - 5):     
            dice_score = evaluate(args, epoch_num, exp_name= args.exp_name)
            logging.info("Validation Dice score at epoch {}, mean_dice:{:.4f}".format(epoch_num, dice_score))
            if dice_score > best_performance:
                # 保存新的最优模型
                save_mode_path = save_mode_path.replace('epoch_' + str(epoch_num) + '.pth', 'best_model.pth')
                best_model_path = save_mode_path
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"New best model saved at epoch {epoch_num} with Dice {dice_score:.4f}")
                best_model_num = epoch_num
                best_performance = dice_score
            if epoch_num >=max_epoch:
                evaluate(args, best_model_num, exp_name= args.exp_name)
                iterator.close()
                break


    writer.close()
    return "Training Finished!"
