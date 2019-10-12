import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_images(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    image_names = [d for d in os.listdir(dir)]
    for image_name in image_names:
        if has_file_allowed_extension(image_name, extensions):
            file = os.path.join(dir, image_name)
            images.append(file)
    return images


#Checks if a file is an allowed extension.
def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def mean_acc(target_indice, pred_indice, num_classes, classes=None):
    assert(num_classes == len(classes))
    acc = 0.
    print('{0} Class Acc Report {1}'.format('#' * 10, '#' * 10))
    for i in range(num_classes):
        idx = np.where(target_indice == i)[0]
        # acc = acc + accuracy_score(target_indice[idx], pred_indice[idx])
        class_correct = accuracy_score(target_indice[idx], pred_indice[idx])
        acc += class_correct
        print('acc {0}: {1:.3f}'.format(classes[i], class_correct * 100))

        # class report
        # y_tpye, y_true, y_pred = _check_targets(target_indice[idx], pred_indice[idx])
        # score = y_true == y_pred
        # wrong_index = np.where(score == False)[0]
        # for j in idx[wrong_index]:
        #     print("Wrong for class [%s]: predicted as: <%s>, image_id--<%s>" %
        #           (int_to_class[i], int_to_class[pred[j]], image_paths[j]))
        #
        # print("[class] %s accuracy is %.3f" % (int_to_class[i], class_correct))
    print('#' * 30)
    return (acc / num_classes) * 100

def process_output(output):
    # Computes the result and argmax index
    pred, index = output.topk(1, 1, largest=True)

    return pred.cpu().float().numpy().flatten(), index.cpu().numpy().flatten()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        if weight:
            weight = torch.FloatTensor(weight).cuda()
        # self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)


    def forward(self, inputs, targets):
        return self.loss(inputs, targets)


def accuracy(preds, label):
    valid = (label > 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / float(valid_sum + 1e-10)
    return acc, valid_sum


def color_label_np(label, ignore=None, dataset=None):
    if dataset == 'cityscapes':
        label_colours = label_colours_cityscapes
    elif dataset == 'sunrgbd':
        label_colours = label_colours_sunrgbd
    colored_label = np.vectorize(lambda x: label_colours[-1] if x == ignore else label_colours[int(x)])
    colored = np.asarray(colored_label(label)).astype(np.float32)
    # colored = colored.squeeze()

    try:
        return colored.transpose([1, 2, 0])
    except ValueError:
        return colored[np.newaxis, ...]


def color_label(label, ignore=None, dataset=None):
    # label = label.data.cpu().numpy()
    if dataset == 'cityscapes':
        label_colours = label_colours_cityscapes
    elif dataset == 'sunrgbd':
        label_colours = label_colours_sunrgbd
    elif dataset == 'ade20k':
        label_colours = label_colours_ade20k
    colored_label = np.vectorize(lambda x: label_colours[-1] if x == ignore else label_colours[int(x)])
    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    try:
        return colored.transpose([1, 0, 2, 3])
    except ValueError:
        return colored[np.newaxis, ...]


label_colours_sunrgbd = [
                 (148, 65, 137), (255, 116, 69), (86, 156, 137),
                 (202, 179, 158), (155, 99, 235), (161, 107, 108),
                 (133, 160, 103), (76, 152, 126), (84, 62, 35),
                 (44, 80, 130), (31, 184, 157), (101, 144, 77),
                 (23, 197, 62), (141, 168, 145), (142, 151, 136),
                 (115, 201, 77), (100, 216, 255), (57, 156, 36),
                 (88, 108, 129), (105, 129, 112), (42, 137, 126),
                 (155, 108, 249), (166, 148, 143), (81, 91, 87),
                 (100, 124, 51), (73, 131, 121), (157, 210, 220),
                 (134, 181, 60), (221, 223, 147), (123, 108, 131),
                 (161, 66, 179), (163, 221, 160), (31, 146, 98),
                 (99, 121, 30), (49, 89, 240), (116, 108, 9),
                 (139, 110, 246), (0, 0, 0)]  # list(-1) for 255

label_colours_cityscapes = [
(128, 64, 128),
(244,35,232),
(70 ,0, 70),
(102,102,156),
(190,153,153),
(153,153,153),
(250,170,30),
(220,220,0),
(107,142,35),
(152,251,152),
(70, 130,180),
(220, 20, 60),
(255, 0, 0),
(0,0,142),
(0,0,70),
(0,60,100),
(0,80,100),
(0,0,230),
(119,11,32),
(0, 0, 0)
]

label_colours_ade20k = [
(120,120,120),
(180,120,120),
(6,230,230),
(80,50,50),
(4,200,3),
(120,120,80),
(140,140,140),
(204,5,255),
(230,230,230),
(4,250,7),
(224,5,255),
(235,255,7),
(150,5,61),
(120,120,70),
(8,255,51),
(255,6,82),
(143,255,140),
(204,255,4),
(255,51,7),
(204,70,3),
(0,102,200),
(61,230,250),
(255,6,51),
(11,102,255),
(255,7,71),
(255,9,224),
(9,7,230),
(220,220,220),
(255,9,92),
(112,9,255),
(8,255,214),
(7,255,224),
(255,184,6),
(10,255,71),
(255,41,10),
(7,255,255),
(224,255,8),
(102,8,255),
(255,61,6),
(255,194,7),
(255,122,8),
(0,255,20),
(255,8,41),
(255,5,153),
(6,51,255),
(235,12,255),
(160,150,20),
(0,163,255),
(140,140,140),
(250,10,15),
(20,255,0),
(31,255,0),
(255,31,0),
(255,224,0),
(153,255,0),
(0,0,255),
(255,71,0),
(0,235,255),
(0,173,255),
(31,0,255),
(11,200,200),
(255,82,0),
(0,255,245),
(0,61,255),
(0,255,112),
(0,255,133),
(255,0,0),
(255,163,0),
(255,102,0),
(194,255,0),
(0,143,255),
(51,255,0),
(0,82,255),
(0,255,41),
(0,255,173),
(10,0,255),
(173,255,0),
(0,255,153),
(255,92,0),
(255,0,255),
(255,0,245),
(255,0,102),
(255,173,0),
(255,0,20),
(255,184,184),
(0,31,255),
(0,255,61),
(0,71,255),
(255,0,204),
(0,255,194),
(0,255,82),
(0,10,255),
(0,112,255),
(51,0,255),
(0,194,255),
(0,122,255),
(0,255,163),
(255,153,0),
(0,255,10),
(255,112,0),
(143,255,0),
(82,0,255),
(163,255,0),
(255,235,0),
(8,184,170),
(133,0,255),
(0,255,92),
(184,0,255),
(255,0,31),
(0,184,255),
(0,214,255),
(255,0,112),
(92,255,0),
(0,224,255),
(112,224,255),
(70,184,160),
(163,0,255),
(153,0,255),
(71,255,0),
(255,0,163),
(255,204,0),
(255,0,143),
(0,255,235),
(133,255,0),
(255,0,235),
(245,0,255),
(255,0,122),
(255,245,0),
(10,190,212),
(214,255,0),
(0,204,255),
(20,0,255),
(255,255,0),
(0,153,255),
(0,41,255),
(0,255,204),
(41,0,255),
(41,255,0),
(173,0,255),
(0,245,255),
(71,0,255),
(122,0,255),
(0,255,184),
(0,92,255),
(184,255,0),
(0,133,255),
(255,214,0),
(25,194,194),
(102,255,0),
(92,0,255),
(0, 0, 0)
]

def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def intersectionAndUnion(output, label, num_classes, ignore_index=255):
    # 'K' classes, output and label sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    output = output.reshape(output.size)
    label = label.reshape(label.size)
    output[np.where(label == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == label)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(num_classes + 1))
    area_output, _ = np.histogram(output, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_output + area_label - area_intersection
    return area_intersection, area_union, area_label
def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    p
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def slide_cal(model, image, classes, crop_size, stride_rate=2/3):

    crop_h, crop_w = crop_size
    batch_size, _, h, w = image.size()
    assert crop_h <= h
    assert crop_w <= w
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(w-crop_w)/stride_w) + 1)

    prediction_crop = torch.zeros(batch_size, classes, h, w).cuda()
    count_crop = torch.zeros(batch_size, 1, h, w).cuda()
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, w)
            s_w = e_w - crop_w
            image_crop = image[:, :, s_h:e_h, s_w:e_w].clone()
            count_crop[:, :, s_h:e_h, s_w:e_w] += 1
            prediction_crop[:, :, s_h:e_h, s_w:e_w] += net_process(model, image_crop)

    prediction_crop /= count_crop
    return prediction_crop


def net_process(model, image):

    with torch.no_grad():
        output = model(source=image, phase='test', out_keys=['cls'], return_losses=False)
    output = output['cls']
    return output