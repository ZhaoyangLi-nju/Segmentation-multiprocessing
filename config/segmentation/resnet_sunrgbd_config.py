from config.default_config import DefaultConfig

class RESNET_SUNRGBD_CONFIG:

    def args(self):
        log_dir = '/home/dudapeng/workspace/summary/'

        ########### Quick Setup ############
        model = 'PSP'
        arch = 'resnet50'
        dataset = 'sunrgbd'
        filters = 'bottleneck'

        task_name = 'content_model_remove_maxpool'
        lr_schedule = 'lambda'  # lambda|step|plateau1
        pretrained = 'place'
        content_pretrained = 'place'
        gpus = '4,5,6,7'  # gpu no. you can add more gpus with comma, e.g., '0,1,2'
        batch_size_train = 4
        batch_size_val = 4

        niter = 5000
        niter_decay = 10000
        niter_total = niter + niter_decay

        no_trans = False  # if True, no translation loss
        target_modal = 'depth'
        # target_modal = 'seg'

        if no_trans:
            loss = ['CLS']
        else:
            loss = ['CLS', 'SEMANTIC']

        unlabeld = False  # True for training with unlabeled data
        evaluate = True  # report mean acc after each epoch
        content_layers = '0,1,2,3,4'  # layer-wise semantic layers, you can change it to better adapt your task
        alpha_content = 0.5
        which_content_net = 'resnet50'

        multi_scale = False
        multi_targets = ['depth']
        # multi_targets = ['seg']
        multi_modal = True
        which_score = 'up'
        norm = 'in'

        resume = False
        resume_path = 'FCN/2019_09_17_13_50_34/FCN_AtoB_5000.pth'

        return {

            'TASK': task_name,
            'MODEL': model,
            'GPU_IDS': gpus,
            'BATCH_SIZE_TRAIN': batch_size_train,
            'BATCH_SIZE_VAL': batch_size_val,
            'PRETRAINED': pretrained,
            'FILTERS': filters,
            'DATASET': dataset,

            'LOG_PATH': log_dir,
            'DATA_DIR': DefaultConfig.ROOT_DIR + '/datasets/vm_data/sunrgbd_seg',

            # MODEL
            'ARCH': arch,
            'SAVE_BEST': True,
            'NO_TRANS': no_trans,
            'LOSS_TYPES': loss,

            #### DATA
            'NUM_CLASSES': 37,
            'UNLABELED': unlabeld,

            # TRAINING / TEST
            'RESUME': resume,
            'RESUME_PATH': resume_path,
            'LR_POLICY': lr_schedule,

            'NITER': niter,
            'NITER_DECAY': niter_decay,
            'NITER_TOTAL': niter_total,
            'FIVE_CROP': False,
            'EVALUATE': evaluate,

            # translation task
            'WHICH_CONTENT_NET': which_content_net,
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,
            'ALPHA_CONTENT': alpha_content,
            'TARGET_MODAL': target_modal,
            'MULTI_SCALE': multi_scale,
            'MULTI_TARGETS': multi_targets,
            'WHICH_SCORE': which_score,
            'MULTI_MODAL': multi_modal,
            'UPSAMPLE_NORM': norm
        }
