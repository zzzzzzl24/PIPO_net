import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import re

class SoftmaxWeightedLoss(nn.Module):
    def __init__(self, num_cls=4):
        super(SoftmaxWeightedLoss, self).__init__()
        self.num_cls = num_cls

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_cls):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor),从右至左，先判断是否属于该类，属于则为True,否则false;
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, output, target):
        target = target.float()
        B, _, H, W = output.size()
        target = self._one_hot_encoder(target)
        for i in range(self.num_cls):
            outputi = output[:, i, :, :]
            targeti = target[:, i, :, :]
            weighted = 1.0 - (torch.sum(targeti, (1, 2)) * 1.0 / torch.sum(target, (1, 2, 3)))
            weighted = torch.reshape(weighted, (-1, 1, 1)).repeat(1, H, W)
            # 计算交叉熵损失
            if i == 0:
                cross_loss = 0
            else:
                cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        # 返回平均损失
        cross_loss = torch.mean(cross_loss)
        return cross_loss

class DomainClsLoss(nn.Module):
    def __init__(self):
        super(DomainClsLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], 'predict & target shape do not match'
        total_loss = self.criterion(predict, target)
        return total_loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor),从右至左，先判断是否属于该类，属于则为True,否则false;
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
            weight[0] = 0.5
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            loss_label = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - loss_label.item())
            loss += loss_label * weight[i]
        print(class_wise_dice)
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    
    elif pred.sum() < 200 and gt.sum() <= 200:
        return 1, 0
    else:
        return 0, 0
    
def Compute_multi_modal_loss(
    share_f: torch.Tensor,           # shape: [B, 3, C, H, W]
    spec_logits: torch.Tensor,       # shape: [B, 3, num_classes]
    mode_Type: torch.Tensor,         # shape: [B, 3]，取值为0/1
    distribution_loss,               # 两个特征图之间的损失函数
    loss_domain_cls                  # 分类损失函数，如 nn.CrossEntropyLoss()
):
    """
    计算模态共享特征的 distribution loss 和 domain-specific classification loss。

    返回：
        dis_shared_loss_total: 跨模态特征对齐损失（可缺失跳过）
        dis_spec_loss_total:   domain 分类损失（仅对存在的模态计算）
    """
    batch_size = mode_Type.size(0)
    dis_shared_loss_total = 0.0
    dis_spec_loss_total = 0.0

    for i in range(batch_size):
        mode = mode_Type[i]            # [3]
        shared_feats = share_f[i]      # [3, C, H, W]
        spec_logit = spec_logits[i]    # [3, num_classes]

        # --- 1. Shared Feature Distribution Loss ---
        dis_shared_loss = 0.0
        if mode[0] == 1 and mode[1] == 1:
            dis_shared_loss += distribution_loss(shared_feats[0], shared_feats[1])
        if mode[1] == 1 and mode[2] == 1:
            dis_shared_loss += distribution_loss(shared_feats[1], shared_feats[2])
        if mode[2] == 1 and mode[0] == 1:
            dis_shared_loss += distribution_loss(shared_feats[2], shared_feats[0])
        dis_shared_loss_total += dis_shared_loss

        # --- 2. Specific Feature Classification Loss ---
        valid_idx = (mode == 1).nonzero(as_tuple=True)[0]  # 有效模态的索引
        if len(valid_idx) > 0:
            valid_spec_logits = spec_logit[valid_idx]  # [有效模态数, num_classes]
            valid_spec_labels = valid_idx.to(device=spec_logits.device)
            dis_spec_loss = loss_domain_cls(valid_spec_logits, valid_spec_labels)
            dis_spec_loss_total += dis_spec_loss

    # 也可选择不除以 batch_size，看具体需求
    # dis_shared_loss_total = dis_shared_loss_total / batch_size
    # dis_spec_loss_total = dis_spec_loss_total / batch_size

    return dis_shared_loss_total, dis_spec_loss_total

def test_single_volume(image, image1, image2, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, mode_Type=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    image1, image2 = image1.squeeze(0).cpu().detach().numpy(), image2.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:  # 3D image
        prediction = np.zeros_like(label)

        for ind in range(image.shape[2]):  # iterate over the third dimension (slice dimension)
            slice = image[:, :, ind]
            slice1 = image1[:, :, ind]
            slice2 = image2[:, :, ind]
            x, y = slice.shape[0], slice.shape[1]

            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # resize image slice
                slice1 = zoom(slice1, (patch_size[0] / x, patch_size[1] / y), order=3)
                slice2 = zoom(slice2, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()  # Add batch and channel dimensions
            input1 = torch.from_numpy(slice1).unsqueeze(0).unsqueeze(0).float().cuda()
            input2 = torch.from_numpy(slice2).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input, input1, input2, mode_Type)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)  # take the class with the highest probability
                out = out.cpu().detach().numpy()

                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # resize prediction back
                else:
                    pred = out
                prediction[:, :, ind] = pred  # Store prediction for this slice
    else:  # If image is already 2D
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        input1 = torch.from_numpy(image1).unsqueeze(0).unsqueeze(0).float().cuda()
        input2 = torch.from_numpy(image2).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input,input1,input2), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    
    if test_save_path is not None:
            img_itk = sitk.GetImageFromArray(image.transpose(2,0,1).astype(np.float32))
            prd_itk = sitk.GetImageFromArray(prediction.transpose(2,0,1).astype(np.float32))
            lab_itk = sitk.GetImageFromArray(label.transpose(2,0,1).astype(np.float32))
            img_itk.SetSpacing((z_spacing, 1, 1))
            prd_itk.SetSpacing((z_spacing, 1, 1))
            lab_itk.SetSpacing((z_spacing, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
            sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")

    metric_list = []
    for i in range(1, classes):# Start from 1 to skip background class (if present)
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
        if i == 3:
            prediction[prediction == 2] = 3
            label[label == 2] = 3
            metric_list.append(calculate_metric_percase(prediction == 3, label == 3))
            prediction[prediction >= 1] = 1
            label[label >= 1] = 1
            metric_list.append(calculate_metric_percase(prediction == 1, label == 1))    
    # for idx, (dice, hd95) in enumerate(metric_list):
    #     print(f"Class {idx + 1} - Dice: {dice:.4f}, HD95: {hd95:.4f}")
    
    return metric_list


def filter_txt_file(input_path, output_path):
    # 定义目标模式
    mode_name = ['bSSFP', 'LGE', 'T2w', 'bSSFP-LGE', 'bSSFP-T2w', 'LGE-T2w', 'bSSFP-LGE-T2w']
    pattern_time = re.compile(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\]\s*")
    
    # 构造正则模式：匹配任何一个模态名
    mode_pattern = re.compile('|'.join(re.escape(mode) for mode in mode_name))
    keep_keywords = ['Mean class', 'Average']

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    filtered_lines = []
    for line in lines:
        # 去掉时间戳（格式类似于 [2024-06-26 15:00:00]）
        line_clean = pattern_time.sub('', line)

        # 判断是否保留该行
        if mode_pattern.search(line_clean) or any(kw in line_clean for kw in keep_keywords):
            filtered_lines.append(line_clean)

    # 写入到新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)
