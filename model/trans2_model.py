import math
import os
import time
from collections import OrderedDict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision

import util.utils as util
from util.average_meter import AverageMeter
from . import networks as networks
from .base_model import BaseModel
from tqdm import tqdm
import cv2
import apex
import torch.distributed as dist

class Trans2Net(BaseModel):

    def __init__(self, cfg, writer=None,BatchNorm=None):
        super(Trans2Net, self).__init__(cfg)
        # if os.path.exists(self.save_dir):
        # util.mkdir(self.save_dir)
        self.phase = cfg.PHASE
        self.trans = not cfg.NO_TRANS
        self.content_model = None
        self.writer = writer
        self.batch_size_train = cfg.BATCH_SIZE_TRAIN
        self.batch_size_val = cfg.BATCH_SIZE_VAL

        # networks
        self.net = networks.define_netowrks(cfg, device=self.device,SyncBatchNorm=BatchNorm)
        # networks.print_network(self.net)

        if 'PSP' in cfg.MODEL:
            self.modules_ori = [self.net.layer0, self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
            self.modules_new = [self.net.ppm, self.net.cls, self.net.aux]

            if self.trans:
                self.modules_ori.extend([self.net.up1, self.net.up2, self.net.up3,
                                         self.net.up4, self.net.up_seg])
            self.params_list = []
            for module in self.modules_new:
                self.params_list.append(dict(params=module.parameters(), lr=cfg.LR * 10))
            for module in self.modules_ori:
                self.params_list.append(dict(params=module.parameters(), lr=cfg.LR))

    def build_output_keys(self, trans=True, cls=True):

        out_keys = []

        if trans:
            out_keys.append('gen_img')

        if cls:
            out_keys.append('cls')

        return out_keys

    def _optimize(self, iters):

        self._forward(iters)
        self.optimizer.zero_grad()
        total_loss = self._construct_loss(iters)
        if self.cfg.multiprocessing_distributed:
            with apex.amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        self.optimizer.step()

    def set_criterion(self, cfg):

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.EVALUATE:
            criterion_segmentation = util.CrossEntropyLoss2d(weight=cfg.CLASS_WEIGHTS_TRAIN,
                                                             ignore_index=cfg.IGNORE_LABEL)
            self.net.set_cls_criterion(criterion_segmentation)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES:
            criterion_content = torch.nn.L1Loss()
            content_model = networks.Content_Model(cfg, criterion_content).to(self.device)
            self.net.set_content_model(content_model)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES:
            criterion_pix2pix = torch.nn.L1Loss()
            self.net.set_pix2pix_criterion(criterion_pix2pix)

    def set_input(self, data):
        self.source_modal = data['image'].to(self.device)
        if 'label' in data.keys():
            self.label = data['label']
            if self.phase == 'train':
                self.label = torch.LongTensor(self.label).to(self.device)

        if self.trans:
            target_modal = data[self.cfg.TARGET_MODAL]

            if isinstance(target_modal, list):
                self.target_modal = list()
                for i, item in enumerate(target_modal):
                    self.target_modal.append(item.to(self.device))
            else:
                # self.target_modal = util.color_label(self.label)
                self.target_modal = target_modal.to(self.device)
        else:
            self.target_modal = None

        if self.cfg.WHICH_DIRECTION == 'BtoA':
            self.source_modal, self.target_modal = self.target_modal, self.source_modal

    def train_parameters(self, cfg):

        assert (self.cfg.LOSS_TYPES)
        self.set_criterion(cfg)
        self.set_optimizer(cfg)
        self.set_log_data(cfg)
        self.set_schedulers(cfg)
        if not cfg.multiprocessing_distributed:
            self.net = nn.DataParallel(self.net).to(self.device)

        train_total_steps = 0
        train_total_iter = 0

        total_epoch = int(cfg.NITER_TOTAL / math.ceil((self.train_image_num / cfg.BATCH_SIZE_TRAIN)))
        print('total epoch:{0}, total iters:{1}'.format(total_epoch, cfg.NITER_TOTAL))

        if cfg.INFERENCE:
            self.evaluate()
            print(
                'MIOU: {miou}, mAcc: {macc}, acc: {acc}'.format(miou=self.loss_meters[
                                                                         'VAL_CLS_MEAN_IOU'].val * 100,
                                                                macc=self.loss_meters[
                                                                         'VAL_CLS_MEAN_ACC'].val * 100,
                                                                acc=self.loss_meters[
                                                                        'VAL_CLS_ACC'].val * 100))
            return

        for epoch in range(cfg.START_EPOCH, total_epoch + 1):
            if train_total_iter > cfg.NITER_TOTAL:
                break
            if cfg.multiprocessing_distributed:####多进程 sample
            	cfg.train_sampler.set_epoch(epoch)
            self.update_learning_rate(step=train_total_iter)

            # current_lr = util.poly_learning_rate(cfg.LR, train_total_iter, cfg.NITER_TOTAL, power=0.8)


            # if cfg.LR_POLICY != 'plateau':
            #     self.update_learning_rate(step=train_total_iter)
            # else:
            #     self.update_learning_rate(val=self.loss_meters['VAL_CLS_LOSS'].avg)

            self.print_lr()

            self.imgs_all = []
            self.pred_index_all = []
            self.target_index_all = []
            self.fake_image_num = 0

            start_time = time.time()

            self.phase = 'train'
            self.net.train()

            for key in self.loss_meters:
                self.loss_meters[key].reset()

            iters = 0

            print('# Training images num = {0}'.format(self.train_image_num))
            # batch = tqdm(self.val_loader, total=self.train_image_num // self.batch_size_train)
            for i, data in enumerate(self.train_loader):
                self.set_input(data)
                train_total_steps += self.batch_size_train
                train_total_iter += 1
                iters += 1
                self._optimize(train_total_iter)
                # self.val_iou = self.validate(train_total_iter)
                # self._write_loss(phase=self.phase, global_step=train_total_iter)

            print('log_path:', cfg.LOG_PATH)
            print('iters in one epoch:', iters)
            print('gpu_ids:', cfg.GPU_IDS)
            self._write_loss(phase=self.phase, global_step=train_total_iter)
            print('Epoch: {epoch}/{total}'.format(epoch=epoch, total=total_epoch))
            train_errors = self.get_current_errors(current=False)
            print('#' * 10)
            self.print_current_errors(train_errors, epoch)
            print('#' * 10)
            print('Training Time: {0} sec'.format(time.time() - start_time))

            # Validate cls
            # or train_total_iter > cfg.NITER_TOTAL * 0.7
            # if cfg.EVALUATE:
            # if cfg.EVALUATE and (epoch % 5 == 0 or epoch == total_epoch):
            # if cfg.EVALUATE and (epoch == total_epoch):

            #     print('# Cls val images num = {0}'.format(self.val_image_num))
            #     self.evaluate()
            #     print(
            #         '{epoch}/{total} MIOU: {miou}, mAcc: {macc}, acc: {acc}'.format(epoch=epoch, total=total_epoch,
            #                                                                         miou=self.loss_meters[
            #                                                                                  'VAL_CLS_MEAN_IOU'].val * 100,
            #                                                                         macc=self.loss_meters[
            #                                                                                  'VAL_CLS_MEAN_ACC'].val * 100,
            #                                                                         acc=self.loss_meters[
            #                                                                                 'VAL_CLS_ACC'].val * 100))
            #     self._write_loss(phase=self.phase, global_step=train_total_iter)

            print('End of iter {0} / {1} \t '
                  'Time Taken: {2} sec'.format(train_total_iter, cfg.NITER_TOTAL, time.time() - start_time))
            print('-' * 80)

    # encoder-decoder branch
    def _forward(self, if_cls=None, if_trans=None):

        self.gen = None
        self.cls_loss = None
        self.source_modal_show = self.source_modal

        if if_cls is None or if_trans is None:

            if self.phase == 'train':

                if 'CLS' not in self.cfg.LOSS_TYPES:
                    if_trans = True
                    if_cls = False

                elif self.trans and 'CLS' in self.cfg.LOSS_TYPES:
                    if_trans = True
                    if_cls = True
                else:
                    if_trans = False
                    if_cls = True
            else:
                if_cls = True
                if_trans = False

        out_keys = self.build_output_keys(trans=if_trans, cls=if_cls)

        self.result = self.net(source=self.source_modal, target=self.target_modal, label=self.label, out_keys=out_keys,
                               phase=self.phase)

        if if_cls:
            self.cls = self.result['cls']

        if if_trans:
            if self.cfg.MULTI_MODAL:
                self.gen = [self.result['gen_img_1'], self.result['gen_img_2']]
            else:
                self.gen = self.result['gen_img']

    def _construct_loss(self, iters):
        if self.cfg.multiprocessing_distributed:
            loss_total = self.result['loss_cls']* self.cfg.ALPHA_CLS
            self.Train_predicted_label = self.cls.data.max(1)[1].cpu().numpy()
            
            return loss_total
        else:
            loss_total = torch.zeros(1)
        if self.use_gpu:
            loss_total = loss_total.to(self.device)

        if 'CLS' in self.cfg.LOSS_TYPES:
            print(self.result['loss_cls'].size())
            cls_loss = self.result['loss_cls'].mean() * self.cfg.ALPHA_CLS
            loss_total = loss_total + cls_loss

            cls_loss = round(cls_loss.item(), 4)
            self.loss_meters['TRAIN_CLS_LOSS'].update(cls_loss)
            # self.Train_predicted_label = self.cls.data
            self.Train_predicted_label = self.cls.data.max(1)[1].cpu().numpy()

        # ) content supervised
        if 'SEMANTIC' in self.cfg.LOSS_TYPES:
            decay_coef = 1
            # decay_coef = (iters / self.cfg.NITER_TOTAL)  # small to big
            # decay_coef = max(0, (self.cfg.NITER_TOTAL - iters) / self.cfg.NITER_TOTAL) # big to small
            content_loss = self.result['loss_content'].mean() * self.cfg.ALPHA_CONTENT * decay_coef
            loss_total = loss_total + content_loss

            content_loss = round(content_loss.item(), 4)
            self.loss_meters['TRAIN_SEMANTIC_LOSS'].update(content_loss)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES:
            decay_coef = 1
            # decay_coef = (iters / self.cfg.NITER_TOTAL)  # small to big
            # decay_coef = max(0, (self.cfg.NITER_TOTAL - iters) / self.cfg.NITER_TOTAL) # big to small
            pix2pix_loss = self.result['loss_pix2pix'].mean() * self.cfg.ALPHA_PIX2PIX * decay_coef
            loss_total = loss_total + pix2pix_loss

            pix2pix_loss = round(pix2pix_loss.item(), 4)
            self.loss_meters['TRAIN_PIX2PIX_LOSS'].update(pix2pix_loss)

        
        return loss_total

    def set_log_data(self, cfg):

        self.loss_meters = defaultdict()
        self.log_keys = [
            'TRAIN_G_LOSS',
            'TRAIN_SEMANTIC_LOSS',  # semantic
            'TRAIN_PIX2PIX_LOSS',
            'TRAIN_CLS_ACC',
            'VAL_CLS_ACC',  # classification
            'TRAIN_CLS_LOSS',
            'TRAIN_CLS_MEAN_IOU',
            'VAL_CLS_LOSS',
            'VAL_CLS_MEAN_IOU',
            'VAL_CLS_MEAN_ACC'
        ]
        for item in self.log_keys:
            self.loss_meters[item] = AverageMeter()

    def save_checkpoint(self, iter, filename=None):

        if filename is None:
            filename = 'Trans2_{0}_{1}.pth'.format(self.cfg.WHICH_DIRECTION, iter)

        net_state_dict = self.net.state_dict()
        save_state_dict = {}
        for k, v in net_state_dict.items():
            if 'content_model' in k:
                continue
            save_state_dict[k] = v

        state = {
            'iter': iter,
            'state_dict': save_state_dict,
            'optimizer': self.optimizer.state_dict(),
        }

        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)

    def load_checkpoint(self, net, checkpoint_path):

        keep_fc = not self.cfg.NO_FC

        if os.path.isfile(checkpoint_path):

            state_checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
            # print(state_checkpoint.keys()[0])
            print('loading model...')
            # load from pix2pix net_G, no cls weights, selected update
            state_dict = net.state_dict()
            # print(state_dict.keys()[0])
            # state_checkpoint = checkpoint
            # state_checkpoint = checkpoint['state_dict']
            state_checkpoint = {str.replace(k, 'module.', ''): v for k, v in state_checkpoint['state_dict'].items()}

            if keep_fc:
                pretrained_G = {k: v for k, v in state_checkpoint.items() if k in state_dict}
            else:
                pretrained_G = {k: v for k, v in state_checkpoint.items() if k in state_dict and 'fc' not in k}

            state_dict.update(pretrained_G)
            net.load_state_dict(state_dict)

            # if self.phase == 'train' and not self.cfg.INIT_EPOCH:
            #     optimizer.load_state_dict(checkpoint['optimizer'])

            # print("=> loaded checkpoint '{}' (iter {})"
            #       .format(checkpoint_path, checkpoint['iter']))
        else:
            print("=> !!! No checkpoint found at '{}'".format(self.cfg.RESUME_PATH))
            return

    def set_optimizer(self, cfg):

        self.optimizers = []
        # self.optimizer = torch.optim.Adam([{'params': self.net.fc.parameters(), 'lr': cfg.LR}], lr=cfg.LR / 10, betas=(0.5, 0.999))

        # if 'PSP' in cfg.MODEL:
        #     self.optimizer = torch.optim.SGD(self.params_list, lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        # else:
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.LR, betas=(0.5, 0.999))

        if 'PSP' in cfg.MODEL:
            self.optimizer = torch.optim.Adam(self.params_list, lr=cfg.LR, betas=(0.5, 0.999))
            # self.optimizer = torch.optim.SGD(self.params_list,lr=cfg.LR, momentum=cfg.MOMENTUM,weight_decay=cfg.WEIGHT_DECAY)

            # for index in range(0, 5):
            #     self.optimizer.param_groups[index]['lr'] = cfg.LR
            # for index in range(5, len(self.optimizer.param_groups)):
            #     self.optimizer.param_groups[index]['lr'] = cfg.LR * 10

        # print('optimizer: ', self.optimizer)
        self.optimizers.append(self.optimizer)
        if cfg.multiprocessing_distributed:
            self.net, self.optimizer = apex.amp.initialize(self.net.cuda(), self.optimizer, opt_level=cfg.opt_level)
            self.net  = apex.parallel.DistributedDataParallel(self.net)

    def evaluate(self):

        if not self.cfg.SLIDE_WINDOWS:
            self.validate()
        else:
            self.test_slide()

    def validate(self):

        self.phase = 'test'

        # switch to evaluate mode
        self.net.eval()

        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        # batch = tqdm(self.val_loader, total=self.val_image_num // self.batch_size_val)
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                self.set_input(data)
                self._forward(if_cls=True, if_trans=False)

                cls_loss = self.result['loss_cls'].mean() * self.cfg.ALPHA_CLS
                self.loss_meters['VAL_CLS_LOSS'].update(cls_loss)
                self.pred = self.cls.data.max(1)[1]
                # self.pred = self.cls.data.max(1)[1].cpu().numpy()
                # label = np.uint8(self.label)

                intersection, union, label = util.intersectionAndUnionGPU(self.pred.cuda(), self.label.cuda(),
                                                                       self.cfg.NUM_CLASSES)
                if self.cfg.multiprocessing_distributed:
                    dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(label)
                intersection, union, label = intersection.cpu().numpy(), union.cpu().numpy(), label.cpu().numpy()
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(label)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        self.loss_meters['VAL_CLS_ACC'].update(allAcc)
        self.loss_meters['VAL_CLS_MEAN_ACC'].update(mAcc)
        self.loss_meters['VAL_CLS_MEAN_IOU'].update(mIoU)

    def test_slide(self):
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.net.eval()
        self.phase = 'test'

        print('testing sliding windows...')
        # batch = tqdm(self.val_loader, total=self.val_image_num // self.batch_size_val)
        for i, data in enumerate(self.val_loader):
            self.set_input(data)
            prediction = util.slide_cal(model=self.net, image=self.source_modal, classes=self.cfg.NUM_CLASSES, crop_size=self.cfg.FINE_SIZE)
            # self.pred = prediction.data.max(1)[1].cpu().numpy()
            self.pred = prediction.max(1)[1]
            # label = np.uint8(self.label)

            intersection, union, label = util.intersectionAndUnionGPU(self.pred.cuda(), self.label.cuda(), self.cfg.NUM_CLASSES)
            if self.cfg.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(label)
            intersection, union, label = intersection.cpu().numpy(), union.cpu().numpy(), label.cpu().numpy()
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(label)
            print(sum(intersection_meter.val)/sum(target_meter.val))

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)


        self.loss_meters['VAL_CLS_ACC'].update(allAcc)
        self.loss_meters['VAL_CLS_MEAN_ACC'].update(mAcc)
        self.loss_meters['VAL_CLS_MEAN_IOU'].update(mIoU)

    def _write_loss(self, phase, global_step):

        loss_types = self.cfg.LOSS_TYPES

        if self.phase == 'train':
            self.label_show = self.label.data.cpu().numpy()
        else:
            self.label_show = np.uint8(self.label)

        self.source_modal_show = self.source_modal
        self.target_modal_show = self.target_modal

        if phase == 'train':

            self.writer.add_scalar('Seg/LR', self.optimizer.param_groups[0]['lr'], global_step=global_step)

            if 'CLS' in loss_types:
                self.writer.add_scalar('Seg/TRAIN_CLS_LOSS', self.loss_meters['TRAIN_CLS_LOSS'].avg,
                                       global_step=global_step)
                # self.writer.add_scalar('TRAIN_CLS_ACC', self.loss_meters['TRAIN_CLS_ACC'].avg*100.0,
                #                        global_step=global_step)
                # self.writer.add_scalar('TRAIN_CLS_MEAN_IOU', float(self.train_iou.mean())*100.0,
                #                        global_step=global_step)

            if self.trans:

                if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                    self.writer.add_scalar('Seg/TRAIN_SEMANTIC_LOSS', self.loss_meters['TRAIN_SEMANTIC_LOSS'].avg,
                                           global_step=global_step)
                if 'PIX2PIX' in self.cfg.LOSS_TYPES:
                    self.writer.add_scalar('Seg/TRAIN_PIX2PIX_LOSS', self.loss_meters['TRAIN_PIX2PIX_LOSS'].avg,
                                           global_step=global_step)

                if isinstance(self.target_modal, list):
                    for i, (gen, target) in enumerate(zip(self.gen, self.target_modal)):
                        self.writer.add_image('Seg/2_Train_Gen_' + str(self.cfg.FINE_SIZE / pow(2, i)),
                                              torchvision.utils.make_grid(gen[:6].clone().cpu().data, 3,
                                                                          normalize=True),
                                              global_step=global_step)
                        self.writer.add_image('Seg/3_Train_Target_' + str(self.cfg.FINE_SIZE / pow(2, i)),
                                              torchvision.utils.make_grid(target[:6].clone().cpu().data, 3,
                                                                          normalize=True),
                                              global_step=global_step)
                else:
                    self.writer.add_image('Seg/Train_target',
                                          torchvision.utils.make_grid(self.target_modal_show[:6].clone().cpu().data, 3,
                                                                      normalize=True), global_step=global_step)
                    self.writer.add_image('Seg/Train_gen',
                                          torchvision.utils.make_grid(self.gen.data[:6].clone().cpu().data, 3,
                                                                      normalize=True), global_step=global_step)

            self.writer.add_image('Seg/Train_image',
                                  torchvision.utils.make_grid(self.source_modal_show[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)
            if 'CLS' in loss_types:
                self.writer.add_image('Seg/Train_predicted',
                                      torchvision.utils.make_grid(
                                          torch.from_numpy(util.color_label(self.Train_predicted_label[:6],
                                                                            ignore=self.cfg.IGNORE_LABEL,
                                                                            dataset=self.cfg.DATASET)), 3,
                                          normalize=True, range=(0, 255)), global_step=global_step)
                # torchvision.utils.make_grid(util.color_label(torch.max(self.Train_predicted_label[:6], 1)[1]+1), 3, normalize=False,range=(0, 255)), global_step=global_step)
                self.writer.add_image('Seg/Train_label',
                                      torchvision.utils.make_grid(
                                          torch.from_numpy(
                                              util.color_label(self.label_show[:6], ignore=self.cfg.IGNORE_LABEL,
                                                               dataset=self.cfg.DATASET)), 3, normalize=True,
                                          range=(0, 255)), global_step=global_step)

        if phase == 'test':
            self.writer.add_image('Seg/Val_image',
                                  torchvision.utils.make_grid(self.source_modal_show[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)

            self.writer.add_image('Seg/Val_predicted',
                                  torchvision.utils.make_grid(
                                      torch.from_numpy(util.color_label(self.pred[:6], ignore=self.cfg.IGNORE_LABEL,
                                                                        dataset=self.cfg.DATASET)), 3,
                                      normalize=True, range=(0, 255)), global_step=global_step)
            # torchvision.utils.make_grid(util.color_label(torch.max(self.val_iou[:3], 1)[1]+1), 3, normalize=False,range=(0, 255)), global_step=global_step)
            self.writer.add_image('Seg/Val_label',
                                  torchvision.utils.make_grid(torch.from_numpy(
                                      util.color_label(self.label_show[:6], ignore=self.cfg.IGNORE_LABEL,
                                                       dataset=self.cfg.DATASET)),
                                      3, normalize=True, range=(0, 255)),
                                  global_step=global_step)

            self.writer.add_scalar('Seg/VAL_CLS_LOSS', self.loss_meters['VAL_CLS_LOSS'].avg,
                                   global_step=global_step)

            self.writer.add_scalar('Seg/VAL_CLS_ACC', self.loss_meters['VAL_CLS_ACC'].val * 100.0,
                                   global_step=global_step)
            self.writer.add_scalar('Seg/VAL_CLS_MEAN_ACC', self.loss_meters['VAL_CLS_MEAN_ACC'].val * 100.0,
                                   global_step=global_step)
            self.writer.add_scalar('Seg/VAL_CLS_MEAN_IOU', self.loss_meters['VAL_CLS_MEAN_IOU'].val * 100.0,
                                   global_step=global_step)


def get_confusion_matrix(gt_label, pred_label, class_num=37):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix
