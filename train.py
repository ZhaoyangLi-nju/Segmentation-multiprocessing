import os
import random
import sys
from datetime import datetime
from functools import reduce

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader  # new add

import util.utils as util
from config.default_config import DefaultConfig
from config.segmentation.resnet_sunrgbd_config import RESNET_SUNRGBD_CONFIG
from config.segmentation.resnet_cityscape_config import RESNET_CITYSCAPE_CONFIG
from config.segmentation.resnet_ade20k_config import RESNET_ADE20K_CONFIG
from data import segmentation_dataset_cv2
from data import segmentation_dataset
from model.trans2_model import Trans2Net
from model.trans2_multimodal import TRans2Multimodal
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
import cv2

def main():
	cfg = DefaultConfig()
	args = {
	    'resnet_sunrgbd': RESNET_SUNRGBD_CONFIG().args(),
	    'resnet_cityscapes': RESNET_CITYSCAPE_CONFIG().args(),
	    'resnet_ade20k': RESNET_ADE20K_CONFIG().args()
	}

	# use shell
	if len(sys.argv) > 1:
	    device_ids = torch.cuda.device_count()
	    print('device_ids:', device_ids)
	    gpu_ids, config_key, arg_model, arg_target, arg_alpha, arg_task, *arg_loss = sys.argv[1:]
	    cfg.parse(args[config_key])

	    cfg.GPU_IDS = gpu_ids
	    cfg.MODEL = arg_model
	    cfg.TARGET_MODAL = arg_target
	    cfg.ALPHA_CONTENT = float(arg_alpha)
	    cfg.LOSS_TYPES = arg_loss
	    cfg.TASK = arg_task

	else:
	    # config_key = 'resnet_sunrgbd'
	    config_key = 'resnet_cityscapes'
	    # config_key = 'resnet_ade20k'
	    cfg.parse(args[config_key])
	    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in cfg.GPU_IDS)

	trans_task = ''
	if not cfg.NO_TRANS:
	    if cfg.MULTI_MODAL:
	        trans_task = trans_task + '_multimodal_'
	    else:
	        trans_task = trans_task + '_to_' + cfg.TARGET_MODAL

	    trans_task = trans_task + '_alpha_' + str(cfg.ALPHA_CONTENT)

	cfg.LOG_PATH = os.path.join(cfg.LOG_PATH, cfg.MODEL, cfg.CONTENT_PRETRAINED,
	                            ''.join(
	                                [cfg.TASK, trans_task, '_', '.'.join(cfg.LOSS_TYPES),
	                                 '_gpus_', str(len(cfg.GPU_IDS))]), datetime.now().strftime('%b%d_%H-%M-%S'))

	# Setting random seed
	if cfg.MANUAL_SEED is None:
	    cfg.MANUAL_SEED = random.randint(1, 10000)
	random.seed(cfg.MANUAL_SEED)
	torch.manual_seed(cfg.MANUAL_SEED)
	torch.backends.cudnn.benchmark = True
	cfg.project_name = reduce(lambda x, y: str(x) + '/' + str(y), os.path.realpath(__file__).split(os.sep)[:-1])
	util.mkdir('logs')
	# dataset = segmentation_dataset
	dataset = segmentation_dataset_cv2
	which_dataset = None
	train_transforms = list()
	val_transforms = list()
	ms_targets = []
	if 'sunrgbd' in config_key:
	    which_dataset = 'SUNRGBD'
	    train_transforms.append(dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)))
	    train_transforms.append(dataset.Crop((cfg.FINE_SIZE, cfg.FINE_SIZE)))
	    train_transforms.append(dataset.RandomHorizontalFlip())
	    # if cfg.MULTI_SCALE:
	    #     for item in cfg.MULTI_SCALE_TARGETS:
	    #         ms_targets.append(item)
	    #     train_transforms.append(dataset.MultiScale(size=(cfg.FINE_SIZE, cfg.FINE_SIZE),
	    #                                                             scale_times=cfg.MULTI_SCALE_NUM, ms_targets=ms_targets))
	    train_transforms.append(dataset.ToTensor(ms_targets=ms_targets))
	    train_transforms.append(
	        dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], ms_targets=ms_targets))

	    val_transforms = list()
	    val_transforms.append(dataset.Resize(size=(cfg.LOAD_SIZE, cfg.LOAD_SIZE)))
	    val_transforms.append(dataset.Crop((cfg.FINE_SIZE, cfg.FINE_SIZE)))
	    val_transforms.append(dataset.ToTensor())
	    val_transforms.append(dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

	elif 'cityscapes' in config_key:

	    which_dataset = 'CityScapes'

	    # train_transforms.append(dataset.Resize(cfg.LOAD_SIZE))
	    # train_transforms.append(dataset.RandomScale(cfg.RANDOM_SCALE_SIZE))  #
	    train_transforms.append(dataset.RandomCrop(cfg.FINE_SIZE))  #
	    # train_transforms.append(dataset.RandomRotate())
	    train_transforms.append(dataset.RandomHorizontalFlip())
	    train_transforms.append(dataset.ToTensor(ms_targets=ms_targets))
	    train_transforms.append(
	        dataset.Normalize(mean=cfg.MEAN, std=cfg.STD, ms_targets=ms_targets))

	    # val_transforms.append(dataset.Resize(cfg.LOAD_SIZE))
	    # val_transforms.append(dataset.CenterCrop(cfg.FINE_SIZE))
	    val_transforms.append(dataset.ToTensor())
	    val_transforms.append(dataset.Normalize(mean=cfg.MEAN, std=cfg.STD))
	elif 'ade20k' in config_key:

	    which_dataset = 'ADE20K'

	    train_transforms.append(dataset.Resize(cfg.LOAD_SIZE))
	    train_transforms.append(dataset.RandomScale(cfg.RANDOM_SCALE_SIZE))  #
	    train_transforms.append(dataset.RandomCrop(cfg.FINE_SIZE))  #
	    # train_transforms.append(dataset.RandomRotate())
	    train_transforms.append(dataset.RandomHorizontalFlip())
	    train_transforms.append(dataset.ToTensor(ms_targets=ms_targets))
	    train_transforms.append(
	        dataset.Normalize(mean=cfg.MEAN, std=cfg.STD, ms_targets=ms_targets))

	    val_transforms.append(dataset.Resize(cfg.LOAD_SIZE))
	    # val_transforms.append(dataset.CenterCrop(cfg.FINE_SIZE))
	    val_transforms.append(dataset.ToTensor())
	    val_transforms.append(dataset.Normalize(mean=cfg.MEAN, std=cfg.STD))


	cfg.train_data = dataset.__dict__[which_dataset](cfg=cfg, transform=transforms.Compose(train_transforms), phase_train=True,
	                                             data_dir=cfg.DATA_DIR)
	cfg.val_data = dataset.__dict__[which_dataset](cfg=cfg, transform=transforms.Compose(val_transforms), phase_train=False,
                                           data_dir=cfg.DATA_DIR)
	
	# cfg.multiprocessing_distributed = False#new add 
	cfg.rank=0
	cfg.ngpus_per_node = len(cfg.GPU_IDS)
	cfg.dist_url='tcp://127.0.0.1:6789'
	cfg.dist_backend= 'nccl'
	cfg.world_size=1
	cfg.opt_level='O0'
	cfg.print_args()
	if cfg.multiprocessing_distributed:
		cv2.ocl.setUseOpenCL(False)
		cv2.setNumThreads(0)
		# torch.backends.cudnn.benchmark = False
		# torch.backends.cudnn.deterministic = True
		cfg.world_size=cfg.ngpus_per_node*cfg.world_size
		mp.spawn(core_work, nprocs=cfg.ngpus_per_node, args=(cfg.ngpus_per_node, cfg))
	else:
		core_work(cfg.GPU_IDS,cfg.ngpus_per_node,cfg)

def core_work(gpu,ngpus_per_node,cfg):

	if cfg.multiprocessing_distributed:
		cfg.gpu=gpu
		cfg.rank=cfg.rank*ngpus_per_node+gpu
		dist.init_process_group(backend=cfg.dist_backend, init_method= cfg.dist_url, world_size=cfg.world_size, rank=cfg.rank)
		cfg.BATCH_SIZE_TRAIN = int(cfg.BATCH_SIZE_TRAIN / ngpus_per_node)
		cfg.BATCH_SIZE_VAL = int(cfg.BATCH_SIZE_VAL / ngpus_per_node)
		cfg.WORKERS = int(cfg.WORKERS / ngpus_per_node)
		print(cfg.BATCH_SIZE_TRAIN)
		torch.cuda.set_device(gpu)
	if cfg.multiprocessing_distributed:

		train_sampler = torch.utils.data.distributed.DistributedSampler(cfg.train_data)
		val_sampler = torch.utils.data.distributed.DistributedSampler(cfg.val_data)
	else:
		train_sampler = None
		val_sampler = None
	cfg.train_sampler=train_sampler
	cfg.gpu=gpu
	train_loader = DataLoader(cfg.train_data, batch_size=cfg.BATCH_SIZE_TRAIN, shuffle=(cfg.train_sampler is None),
	                          num_workers=cfg.WORKERS, pin_memory=True, drop_last=True, sampler=cfg.train_sampler)

	val_loader = DataLoader(cfg.val_data, batch_size=cfg.BATCH_SIZE_VAL, shuffle=False,
	                        num_workers=cfg.WORKERS,pin_memory=True, sampler=val_sampler)

	unlabeled_loader = None
	num_train = len(cfg.train_data)
	num_val = len(cfg.val_data)

	cfg.CLASS_WEIGHTS_TRAIN = cfg.train_data.class_weights
	cfg.IGNORE_LABEL = cfg.train_data.ignore_label

	# shell script to run1
	print('LOSS_TYPES:', cfg.LOSS_TYPES)
	writer = SummaryWriter(log_dir=cfg.LOG_PATH)  # tensorboard

	if cfg.MULTI_MODAL:
	    model = TRans2Multimodal(cfg, writer=writer)
	else:
		if cfg.multiprocessing_distributed:
			SyncBatchNorm=apex.parallel.SyncBatchNorm
		else:
			from lib.sync_bn.modules import BatchNorm2d as SyncBatchNorm
		model = Trans2Net(cfg, writer=writer,BatchNorm=SyncBatchNorm)
	model.set_data_loader(train_loader, val_loader)



	# def train():
	if cfg.RESUME:
	    checkpoint_path = os.path.join(cfg.CHECKPOINTS_DIR, cfg.RESUME_PATH)
	    # checkpoint = torch.load(checkpoint_path)

	    # load_epoch = checkpoint['epoch']
	    model.load_checkpoint(model.net, checkpoint_path)
	    # cfg.START_EPOCH = load_epoch

	    if cfg.INIT_EPOCH:
	        # just load pretrained parameters
	        print('load checkpoint from another source')
	        cfg.START_EPOCH = 1

	print('>>> task path is {0}'.format(cfg.project_name))

	# train
	model.train_parameters(cfg)

	print('save model ...')
	model_filename = '{0}_{1}_{2}.pth'.format(cfg.MODEL, cfg.WHICH_DIRECTION, cfg.NITER_TOTAL)
	model.save_checkpoint(cfg.NITER_TOTAL, model_filename)

	if writer is not None:
	    writer.close()


if __name__ == '__main__':
    main()
