import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .resnet import resnet18
from .resnet import resnet50
from .resnet import resnet101
# import torchvision.models as models
import model.resnet as models
# from lib.sync_bn.modules import BatchNorm2d as SyncBatchNorm


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def fix_grad(net):
    print(net.__class__.__name__)

    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1:
            m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False

    net.apply(fix_func)


def unfix_grad(net):
    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1 or classname.find('Linear') != -1:
            m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True

    net.apply(fix_func)


def define_netowrks(cfg, device=None,SyncBatchNorm=None):
    if 'resnet' in cfg.ARCH:
        if cfg.MULTI_SCALE:
            model = FCN_Conc_Multiscale(cfg, device=device)
        elif cfg.MULTI_MODAL:
            # model = FCN_Conc_MultiModalTarget_Conc(cfg, device=device)
            model = FCN_Conc_MultiModalTarget(cfg, device=device)
            # model = FCN_Conc_MultiModalTarget_Late(cfg, device=device)
        else:
            if cfg.MODEL == 'FCN':
                model = FCN_Conc(cfg, device=device)
            if cfg.MODEL == 'FCN_MAXPOOL':
                model = FCN_Conc_Maxpool(cfg, device=device)
            # if cfg.MODEL == 'FCN_LAT':
            #     model = FCN_Conc_Lat(cfg, device=device)
            elif cfg.MODEL == 'UNET':
                model = UNet(cfg, device=device)
            # elif cfg.MODEL == 'UNET_256':
            #     model = UNet_Share_256(cfg, device=device)
            # elif cfg.MODEL == 'UNET_128':
            #     model = UNet_Share_128(cfg, device=device)
            # elif cfg.MODEL == 'UNET_64':
            #     model = UNet_Share_64(cfg, device=device)
            # elif cfg.MODEL == 'UNET_LONG':
            #     model = UNet_Long(cfg, device=device)
            elif cfg.MODEL == "PSP":
                model = PSPNet(cfg, BatchNorm=SyncBatchNorm, device=device)
                # model = PSPNet(cfg, BatchNorm=nn.BatchNorm2d, device=device)

    return model


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv_norm_relu(dim_in, dim_out, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d,
                   use_leakyRelu=False, use_bias=False, is_Sequential=True):
    if use_leakyRelu:
        act = nn.LeakyReLU(0.2, True)
    else:
        act = nn.ReLU(True)

    if is_Sequential:
        result = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=use_bias),
            norm(dim_out, affine=True),
            act
        )
        return result
    return [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            norm(dim_out, affine=True),
            act]


def expand_Conv(module, in_channels):
    def expand_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.in_channels = in_channels
            m.out_channels = m.out_channels
            mean_weight = torch.mean(m.weight, dim=1, keepdim=True)
            m.weight.data = mean_weight.repeat(1, in_channels, 1, 1).data

    module.apply(expand_func)


##############################################################################
# Moduels
##############################################################################
# class Upsample_Interpolate(nn.Module):
#
#     def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, norm=nn.BatchNorm2d, scale=2, mode='bilinear'):
#         super(Upsample_Interpolate, self).__init__()
#         self.scale = scale
#         self.mode = mode
#         self.conv_norm_relu1 = conv_norm_relu(dim_in, dim_out, kernel_size=kernel_size, stride=1, padding=padding,
#                                               norm=norm)
#         self.conv_norm_relu2 = conv_norm_relu(dim_out, dim_out, kernel_size=3, stride=1, padding=1, norm=norm)
#
#     def forward(self, x, activate=True):
#         x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
#         x = self.conv_norm_relu1(x)
#         x = self.conv_norm_relu2(x)
#         return x


# class UpConv_Conc(nn.Module):
#
#     def __init__(self, dim_in, dim_out, scale=2, mode='bilinear', norm=nn.BatchNorm2d, if_conc=True):
#         super(UpConv_Conc, self).__init__()
#         self.scale = scale
#         self.mode = mode
#         self.up = nn.Sequential(
#             nn.UpsamplingBilinear2d(scale_factor=scale),
#             conv_norm_relu(dim_in, dim_out, kernel_size=1, padding=0, norm=norm)
#             # nn.Conv2d(dim_in, dim_out, 1, bias=False),
#         )
#         if if_conc:
#             dim_in = dim_out * 2
#         self.conc = nn.Sequential(
#             conv_norm_relu(dim_in, dim_out, kernel_size=1, padding=0, norm=norm),
#             conv_norm_relu(dim_out, dim_out, kernel_size=3, padding=1, norm=norm)
#         )
#
#     def forward(self, x, y=None):
#         x = self.up(x)
#         residual = x
#         if y is not None:
#             x = torch.cat((x, y), 1)
#         return self.conc(x) + residual


# class UpsampleBasicBlock(nn.Module):
#
#     def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d, scale=2, mode='bilinear', upsample=True):
#         super(UpsampleBasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
#                       padding=padding, bias=False)
#         self.bn1 = norm(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm(planes)
#
#         if inplanes != planes:
#             kernel_size, padding = 1, 0
#         else:
#             kernel_size, padding = 3, 1
#
#         if upsample:
#
#             self.trans = nn.Sequential(
#                 nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
#                           padding=padding, bias=False),
#                 norm(planes))
#         else:
#             self.trans = None
#
#         self.scale = scale
#         self.mode = mode
#
#     def forward(self, x):
#
#         if self.trans is not None:
#             x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
#             residual = self.trans(x)
#         else:
#             residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual
#
#         return out

class Conc_Up_Residual(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d, conc_feat=True):
        super(Conc_Up_Residual, self).__init__()

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1,
                      padding=0, bias=False),
            norm(dim_out))
        self.smooth = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,
                                padding=1, bias=False)

        if conc_feat:
            dim_in = dim_out * 2
            kernel_size, padding = 1, 0
        else:
            kernel_size, padding = 3, 1

        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.norm1 = norm(dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_out, dim_out)
        self.norm2 = norm(dim_out)

    def forward(self, x, y=None):

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.smooth(x)
        residual = self.residual_conv(x)

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)

        x += residual

        return self.relu(x)


class Conc_Up_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, conc_feat=True):
        super(Conc_Up_Residual_bottleneck, self).__init__()

        self.smooth = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1,
                                padding=0, bias=False)

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            norm(dim_out))

        if conc_feat:
            dim_in = dim_out * 2
        else:
            dim_in = dim_out

        dim_med = int(dim_out / 2)
        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.smooth(x)
        residual = self.residual_conv(x)

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        x += residual

        return self.relu(x)


class Conc_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, conc_feat=True):
        super(Conc_Residual_bottleneck, self).__init__()

        self.conv0 = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            norm(dim_out))
        # else:
        #     self.residual_conv = nn.Sequential(
        #         nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2,
        #                   padding=1, bias=False),
        #         norm(dim_out))

        if conc_feat:
            dim_in = dim_out * 2
        dim_med = int(dim_out / 2)
        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):

        x = self.conv0(x)
        residual = self.residual_conv(x)

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        x += residual

        return self.relu(x)


class Lat_Up_Residual(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d):
        super(Lat_Up_Residual, self).__init__()

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1,
                      padding=0, bias=False),
            norm(dim_out))
        self.smooth = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,
                                padding=1, bias=False)

        self.lat = nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1,
                             padding=0, bias=False)

        self.conv1 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.norm1 = norm(dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.norm2 = norm(dim_out)

    def forward(self, x, y=None):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.smooth(x)
        residual = self.residual_conv(x)

        if y is not None:
            x = x + self.lat(y)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)

        x += residual

        return self.relu(x)


#########################################

##############################################################################
# Translate to recognize
##############################################################################
class Content_Model(nn.Module):

    def __init__(self, cfg, criterion=None, in_channel=3):
        super(Content_Model, self).__init__()
        self.cfg = cfg
        self.criterion = criterion
        self.net = cfg.WHICH_CONTENT_NET

        if 'resnet' in self.net:
            from .pretrained_resnet import ResNet
            self.model = ResNet(self.net, cfg, in_channel=in_channel)

        fix_grad(self.model)
        # print_network(self)

    def forward(self, x, target, layers=None):

        # important when set content_model as the attr of trecg_net
        self.model.eval()

        layers = layers
        if layers is None or not layers:
            layers = self.cfg.CONTENT_LAYERS.split(',')

        input_features = self.model((x + 1) / 2, layers)
        target_targets = self.model((target + 1) / 2, layers)
        len_layers = len(layers)
        loss_fns = [self.criterion] * len_layers
        alpha = [1] * len_layers

        content_losses = [alpha[i] * loss_fns[i](gen_content, target_targets[i])
                          for i, gen_content in enumerate(input_features)]
        loss = sum(content_losses)
        return loss


# class FCN_Lat(nn.Module):
#
#     def __init__(self, cfg, device=None):
#         super(FCN_Lat, self).__init__()
#
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#         encoder = cfg.ARCH
#         num_classes = cfg.NUM_CLASSES
#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         if cfg.PRETRAINED == 'place':
#             resnet = models.__dict__['resnet18'](num_classes=365)
#             load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
#             checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
#             state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
#             resnet.load_state_dict(state_dict)
#             print('place resnet18 loaded....')
#         else:
#             resnet = resnet18(pretrained=pretrained)
#             print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#         self.head = _FCNHead(512, num_classes, nn.BatchNorm2d)
#         # self.head = nn.Conv2d(512, num_classes, 1)
#
#         if self.using_semantic_branch:
#             self.build_upsample_content_layers(dims, num_classes)
#
#         self.score_aux1 = nn.Conv2d(256, num_classes, 1)
#         self.score_aux2 = nn.Conv2d(128, num_classes, 1)
#         self.score_aux3 = nn.Conv2d(64, num_classes, 1)
#
#         # self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
#         # self.fc = nn.Linear(fc_input_nc, cfg.NUM_CLASSES)
#
#         if pretrained:
#             init_weights(self.head, 'normal')
#
#             if self.trans:
#                 init_weights(self.lat1, 'normal')
#                 init_weights(self.lat2, 'normal')
#                 init_weights(self.lat3, 'normal')
#                 init_weights(self.up1, 'normal')
#                 init_weights(self.up2, 'normal')
#                 init_weights(self.up3, 'normal')
#                 init_weights(self.up4, 'normal')
#
#             init_weights(self.head, 'normal')
#             init_weights(self.score_aux3, 'normal')
#             init_weights(self.score_aux2, 'normal')
#             init_weights(self.score_aux1, 'normal')
#
#         else:
#
#             init_weights(self, 'normal')
#
#     def set_content_model(self, content_model):
#         self.content_model = content_model
#
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def build_upsample_content_layers(self, dims, num_classes):
#
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#
#         self.up1 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm, conc_feat=False)
#         self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm, conc_feat=False)
#         self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)
#         self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
#
#         self.lat1 = nn.Conv2d(dims[3], dims[3], kernel_size=1, stride=1, padding=0, bias=False)
#         self.lat2 = nn.Conv2d(dims[2], dims[2], kernel_size=1, stride=1, padding=0, bias=False)
#         self.lat3 = nn.Conv2d(dims[1], dims[1], kernel_size=1, stride=1, padding=0, bias=False)
#
#         self.up_image_content = nn.Sequential(
#             nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#
#         self.score_up_256 = nn.Sequential(
#             nn.Conv2d(256, num_classes, 1)
#         )
#
#         self.score_up_128 = nn.Sequential(
#             nn.Conv2d(128, num_classes, 1)
#         )
#         self.score_up_64 = nn.Sequential(
#             nn.Conv2d(64, num_classes, 1)
#         )
#
#     def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
#                 return_losses=True):
#         result = {}
#
#         layer_0 = self.relu(self.bn1(self.conv1(source)))
#         if not self.trans:
#             layer_0 = self.maxpool(layer_0)
#         layer_1 = self.layer1(layer_0)
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#
#         if self.trans:
#             # content model branch
#             skip_1 = self.lat1(layer_3)
#             skip_2 = self.lat2(layer_2)
#             skip_3 = self.lat3(layer_1)
#
#             up1 = self.up1(layer_4)
#             up2 = self.up2(up1 + skip_1)
#             up3 = self.up3(up2 + skip_2)
#             up4 = self.up4(up3 + skip_3)
#
#             result['gen_img'] = self.up_image_content(up4)
#             if phase == 'train':
#                 result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)
#
#         if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:
#             # segmentation branch
#             score_head = self.head(layer_4)
#
#             if self.cfg.WHICH_SCORE == 'main' or not self.trans:
#                 score_aux1 = self.score_aux1(layer_3)
#                 score_aux2 = self.score_aux2(layer_2)
#                 score_aux3 = self.score_aux3(layer_1)
#             elif self.cfg.WHICH_SCORE == 'up':
#                 score_aux1 = self.score_aux1(up1)
#                 score_aux2 = self.score_aux2(up2)
#                 score_aux3 = self.score_aux3(up3)
#             elif self.cfg.WHICH_SCORE == 'both':
#                 score_aux1 = self.score_aux1(up1 + layer_3)
#                 score_aux2 = self.score_aux2(up2 + layer_2)
#                 score_aux3 = self.score_aux3(up3 + layer_1)
#
#             score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux1
#             score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux2
#             score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux3
#
#             result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
#
#             if phase == 'train':
#                 result['loss_cls'] = self.cls_criterion(result['cls'], label)
#
#         return result


class FCN_Conc(nn.Module):

    def __init__(self, cfg, device=None):
        super(FCN_Conc, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.resnet18()(num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = models.__dict__[cfg.ARCH](pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        if self.trans:
            self.build_upsample_content_layers(dims)

        if 'resnet18' == cfg.ARCH:
            head_dim = 512
            aux_dims = [256, 128, 64]
        elif 'resnet50' == cfg.ARCH:
            head_dim = 2048
            aux_dims = [1024, 512, 256]

        self.head = _FCNHead(head_dim, num_classes, nn.BatchNorm2d)

        self.score_aux1 = nn.Sequential(
            nn.Conv2d(aux_dims[0], num_classes, 1)
        )

        self.score_aux2 = nn.Sequential(
            nn.Conv2d(aux_dims[1], num_classes, 1)
        )
        self.score_aux3 = nn.Sequential(
            nn.Conv2d(aux_dims[2], num_classes, 1)
        )

        if pretrained:
            init_weights(self.head, 'normal')

            if self.trans:
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')

            init_weights(self.head, 'normal')
            init_weights(self.score_aux3, 'normal')
            init_weights(self.score_aux2, 'normal')
            init_weights(self.score_aux1, 'normal')

        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        
        if 'resnet18' == self.cfg.ARCH:
            self.up1 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
            self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
    
        elif 'resnet50' in self.cfg.ARCH:
            self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
            self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
            self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[1], norm=norm, conc_feat=False)

        self.up_image_content = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}

        layer_0 = self.relu(self.bn1(self.conv1(source)))
        if not self.trans:
            layer_0 = self.maxpool(layer_0)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.trans:
            # translation branch

            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3)

            result['gen_img'] = self.up_image_content(up4)

            if phase == 'train' and 'SEMANTIC' in self.cfg.LOSS_TYPES:
                result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:

            # segmentation branch
            score_head = self.head(layer_4)

            score_aux1 = None
            score_aux2 = None
            score_aux3 = None
            if self.cfg.WHICH_SCORE == 'main' or not self.trans:
                score_aux1 = self.score_aux1(layer_3)
                score_aux2 = self.score_aux2(layer_2)
                score_aux3 = self.score_aux3(layer_1)
            elif self.cfg.WHICH_SCORE == 'up':
                score_aux1 = self.score_aux1(up1)
                score_aux2 = self.score_aux2(up2)
                score_aux3 = self.score_aux3(up3)

            score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux1
            score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux2
            score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux3

            result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            if phase == 'train':
                result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


# class FCN_Conc_Resnet50(nn.Module):
# 
#     def __init__(self, cfg, device=None):
#         super(FCN_Conc_Resnet50, self).__init__()
# 
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#         encoder = cfg.ARCH
#         num_classes = cfg.NUM_CLASSES
# 
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
# 
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
# 
#         # models.BatchNorm = SyncBatchNorm
#         resnet = models.__dict__[cfg.ARCH](pretrained=pretrained, deep_base=False)
#         print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
# 
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#         self.head = _FCNHead(2048, num_classes, nn.BatchNorm2d)
#         # self.head = nn.Conv2d(512, num_classes, 1)
# 
#         if self.trans:
#             self.build_upsample_content_layers(dims)
# 
#         self.score_1024 = nn.Sequential(
#             nn.Conv2d(dims[5], num_classes, 1)
#         )
#         self.score_head = nn.Sequential(
#             nn.Conv2d(dims[4], num_classes, 1)
#         )
#         self.score_aux1 = nn.Sequential(
#             nn.Conv2d(dims[3], num_classes, 1)
#         )
# 
#         if pretrained:
#             init_weights(self.head, 'normal')
# 
#             if self.trans:
#                 init_weights(self.up1, 'normal')
#                 init_weights(self.up2, 'normal')
#                 init_weights(self.up3, 'normal')
#                 init_weights(self.up4, 'normal')
# 
#             init_weights(self.head, 'normal')
#             init_weights(self.score_1024, 'normal')
#             init_weights(self.score_aux1, 'normal')
#             init_weights(self.score_head, 'normal')
# 
#         else:
# 
#             init_weights(self, 'normal')
# 
#     def set_content_model(self, content_model):
#         self.content_model = content_model
# 
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
# 
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
# 
#     def build_upsample_content_layers(self, dims):
# 
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
# 
#         self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
#         self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
#         self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
#         self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[1], norm=norm, conc_feat=False)
# 
#         self.up_image_content = nn.Sequential(
#             nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
# 
#     def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
#                 return_losses=True):
#         result = {}
# 
#         layer_0 = self.relu(self.bn1(self.conv1(source)))
#         if not self.trans:
#             layer_0 = self.maxpool(layer_0)
#         layer_1 = self.layer1(layer_0)
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
# 
#         if self.trans:
#             # content model branch
# 
#             # up1 = self.up1(layer_4)
#             up1 = self.up1(layer_4, layer_3)
#             up2 = self.up2(up1, layer_2)
#             up3 = self.up3(up2, layer_1)
#             up4 = self.up4(up3)
# 
#             result['gen_img'] = self.up_image_content(up4)
# 
#         # segmentation branch
#         # score_2048 = nn.Conv2d(2048, self.cfg.NUM_CLASSES, 1)
#         score_2048 = self.head(layer_4)
# 
#         score_1024 = None
#         score_head = None
#         score_aux1 = None
#         if self.cfg.WHICH_SCORE == 'main' or not self.trans:
#             score_1024 = self.score_1024(layer_3)
#             score_head = self.score_head(layer_2)
#             score_aux1 = self.score_aux1(layer_1)
#         elif self.cfg.WHICH_SCORE == 'up':
#             score_1024 = self.score_1024(up1)
#             score_head = self.score_head(up2)
#             score_aux1 = self.score_aux1(up3)
# 
#         score = F.interpolate(score_2048, score_1024.size()[2:], mode='bilinear', align_corners=True)
#         score = score + score_1024
#         score = F.interpolate(score, score_head.size()[2:], mode='bilinear', align_corners=True)
#         score = score + score_head
#         score = F.interpolate(score, score_aux1.size()[2:], mode='bilinear', align_corners=True)
#         score = score + score_aux1
# 
#         result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
# 
#         if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)
# 
#         if 'CLS' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_cls'] = self.cls_criterion(result['cls'], label)
# 
#         if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)
# 
#         return result


class FCN_Conc_Maxpool(nn.Module):

    def __init__(self, cfg, device=None):
        super(FCN_Conc_Maxpool, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = models.__dict__[cfg.ARCH](pretrained=pretrained, deep_base=False)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        # self.head = nn.Conv2d(512, num_classes, 1)

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                # m.stride = (1, 1)
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        if self.trans:
            self.build_upsample_content_layers(dims)

        if 'resnet18' == cfg.ARCH:
            aux_dims = [256, 128, 64]
            head_dim = 512
        elif 'resnet50' == cfg.ARCH:
            aux_dims = [1024, 512, 256]
            head_dim = 2048

        self.head = _FCNHead(head_dim, num_classes, nn.BatchNorm2d)

        self.score_aux1 = nn.Sequential(
            nn.Conv2d(aux_dims[0], num_classes, 1)
        )

        self.score_aux2 = nn.Sequential(
            nn.Conv2d(aux_dims[1], num_classes, 1)
        )
        self.score_aux3 = nn.Sequential(
            nn.Conv2d(aux_dims[2], num_classes, 1)
        )

        if pretrained:
            init_weights(self.head, 'normal')

            if self.trans:
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')

            init_weights(self.head, 'normal')
            init_weights(self.score_aux3, 'normal')
            init_weights(self.score_aux2, 'normal')
            init_weights(self.score_aux1, 'normal')

        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        if 'resnet18' == self.cfg.ARCH:
            self.up1 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
            self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)

        elif 'resnet50' in self.cfg.ARCH:
            self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
            self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
            self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[1], norm=norm, conc_feat=False)

        self.up_image_content = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}

        layer_0 = self.relu(self.bn1(self.conv1(source)))
        maxpool = self.maxpool(layer_0)
        layer_1 = self.layer1(maxpool)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.trans:
            # content model branch

            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3)

            result['gen_img'] = self.up_image_content(up4)

        # segmentation branch
        score_head = self.head(layer_4)

        score_aux1 = None
        score_aux2 = None
        score_aux3 = None
        if self.cfg.WHICH_SCORE == 'main' or not self.trans:
            score_aux1 = self.score_aux1(layer_3)
            score_aux2 = self.score_aux2(layer_2)
            score_aux3 = self.score_aux3(layer_1)
        elif self.cfg.WHICH_SCORE == 'up':
            score_aux1 = self.score_aux1(up1)
            score_aux2 = self.score_aux2(up2)
            score_aux3 = self.score_aux3(up3)

        score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux1
        score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux2
        score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux3

        result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

        return result


class FCN_Conc_MultiModalTarget(nn.Module):

    def __init__(self, cfg, device=None):
        super(FCN_Conc_MultiModalTarget, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.head = _FCNHead(512, num_classes, nn.BatchNorm2d)
        # self.head = nn.Conv2d(512, num_classes, 1)

        if self.trans:
            self.build_upsample_content_layers(dims)

        self.score_aux1 = nn.Sequential(
            nn.Conv2d(dims[3] * 2, num_classes, 1)
        )

        self.score_aux2 = nn.Sequential(
            nn.Conv2d(dims[2] * 2, num_classes, 1)
        )
        self.score_aux3 = nn.Sequential(
            nn.Conv2d(dims[1] * 2, num_classes, 1)
        )

        # self.score_aux1_depth = nn.Conv2d(256, num_classes, 1)
        # self.score_aux2_depth = nn.Conv2d(128, num_classes, 1)
        # self.score_aux3_depth = nn.Conv2d(64, num_classes, 1)
        #
        # self.score_aux1_seg = nn.Conv2d(256, num_classes, 1)
        # self.score_aux2_seg = nn.Conv2d(128, num_classes, 1)
        # self.score_aux3_seg = nn.Conv2d(64, num_classes, 1)

        # self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
        # self.fc = nn.Linear(fc_input_nc, cfg.NUM_CLASSES)

        if pretrained:
            init_weights(self.head, 'normal')

            if self.trans:
                init_weights(self.up1_depth, 'normal')
                init_weights(self.up2_depth, 'normal')
                init_weights(self.up3_depth, 'normal')
                init_weights(self.up4_depth, 'normal')
                init_weights(self.up1_seg, 'normal')
                init_weights(self.up2_seg, 'normal')
                init_weights(self.up3_seg, 'normal')
                init_weights(self.up4_seg, 'normal')

                # init_weights(self.score_aux3_depth, 'normal')
                # init_weights(self.score_aux2_depth, 'normal')
                # init_weights(self.score_aux1_depth, 'normal')
                # init_weights(self.score_aux3_seg, 'normal')
                # init_weights(self.score_aux2_seg, 'normal')
                # init_weights(self.score_aux1_seg, 'normal')

            init_weights(self.score_aux3, 'normal')
            init_weights(self.score_aux2, 'normal')
            init_weights(self.score_aux1, 'normal')
            init_weights(self.head, 'normal')

        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        if 'bottleneck' in self.cfg.FILTERS:
            self.up1_depth = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up2_depth = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            self.up3_depth = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
            self.up4_depth = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)

            self.up1_seg = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
            self.up2_seg = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
            self.up3_seg = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
            self.up4_seg = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
        else:
            self.up1_depth = Conc_Up_Residual(dims[4], dims[3], norm=norm)
            self.up2_depth = Conc_Up_Residual(dims[3], dims[2], norm=norm)
            self.up3_depth = Conc_Up_Residual(dims[2], dims[1], norm=norm)
            self.up4_depth = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

            self.up1_seg = Conc_Up_Residual(dims[4], dims[3], norm=norm)
            self.up2_seg = Conc_Up_Residual(dims[3], dims[2], norm=norm)
            self.up3_seg = Conc_Up_Residual(dims[2], dims[1], norm=norm)
            self.up4_seg = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

        self.up_depth = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        self.up_seg = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target_1=None, target_2=None, label=None, out_keys=None, phase='train',
                content_layers=None,
                return_losses=True):
        result = {}
        layer_0 = self.relu(self.bn1(self.conv1(source)))
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.trans:
            # content model branch
            up1_depth = self.up1_depth(layer_4, layer_3)
            up2_depth = self.up2_depth(up1_depth, layer_2)
            up3_depth = self.up3_depth(up2_depth, layer_1)
            up4_depth = self.up4_depth(up3_depth)
            result['gen_depth'] = self.up_depth(up4_depth)

            up1_seg = self.up1_seg(layer_4, layer_3)
            up2_seg = self.up2_seg(up1_seg, layer_2)
            up3_seg = self.up3_seg(up2_seg, layer_1)
            up4_seg = self.up4_seg(up3_seg)
            result['gen_seg'] = self.up_seg(up4_seg)

        # segmentation branch
        score_head = self.head(layer_4)

        score_aux1 = None
        score_aux2 = None
        score_aux3 = None
        if self.cfg.WHICH_SCORE == 'main':
            score_aux1 = self.score_aux1(layer_3)
            score_aux2 = self.score_aux2(layer_2)
            score_aux3 = self.score_aux3(layer_1)
        elif self.cfg.WHICH_SCORE == 'up':
            # score_aux1_depth = self.score_aux1_depth(up1_depth)
            # score_aux2_depth = self.score_aux2_depth(up2_depth)
            # score_aux3_depth = self.score_aux3_depth(up3_depth)
            #
            # score_aux1_seg = self.score_aux1_seg(up1_seg)
            # score_aux2_seg = self.score_aux2_seg(up2_seg)
            # score_aux3_seg = self.score_aux3_seg(up3_seg)

            score_aux1 = self.score_aux1(torch.cat((up1_depth, up1_seg), 1))
            score_aux2 = self.score_aux2(torch.cat((up2_depth, up2_seg), 1))
            score_aux3 = self.score_aux3(torch.cat((up3_depth, up3_seg), 1))

        score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux1
        score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux2
        score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux3

        result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_content_depth'] = self.content_model(result['gen_depth'], target_1, layers=content_layers)
            result['loss_content_seg'] = self.content_model(result['gen_seg'], target_2, layers=content_layers)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_pix2pix_depth'] = self.pix2pix_criterion(result['gen_depth'], target_1)
            result['loss_pix2pix_seg'] = self.pix2pix_criterion(result['gen_seg'], target_2)

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class FCN_Conc_MultiModalTarget_Late(nn.Module):

    def __init__(self, cfg, device=None):
        super(FCN_Conc_MultiModalTarget_Late, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.head = _FCNHead(512, num_classes, nn.BatchNorm2d)
        # self.head = nn.Conv2d(512, num_classes, 1)

        if self.trans:
            self.build_upsample_content_layers(dims)

        self.score_aux1 = nn.Sequential(
            nn.Conv2d(dims[3], num_classes, 1)
        )

        self.score_aux2 = nn.Sequential(
            nn.Conv2d(dims[2], num_classes, 1)
        )
        self.score_aux3 = nn.Sequential(
            nn.Conv2d(dims[1], num_classes, 1)
        )

        # self.score_aux1_depth = nn.Conv2d(256, num_classes, 1)
        # self.score_aux2_depth = nn.Conv2d(128, num_classes, 1)
        # self.score_aux3_depth = nn.Conv2d(64, num_classes, 1)
        #
        # self.score_aux1_seg = nn.Conv2d(256, num_classes, 1)
        # self.score_aux2_seg = nn.Conv2d(128, num_classes, 1)
        # self.score_aux3_seg = nn.Conv2d(64, num_classes, 1)

        # self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
        # self.fc = nn.Linear(fc_input_nc, cfg.NUM_CLASSES)

        if pretrained:
            init_weights(self.head, 'normal')

            if self.trans:
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')
                init_weights(self.up_depth, 'normal')
                init_weights(self.up_seg, 'normal')

                # init_weights(self.score_aux3_depth, 'normal')
                # init_weights(self.score_aux2_depth, 'normal')
                # init_weights(self.score_aux1_depth, 'normal')
                # init_weights(self.score_aux3_seg, 'normal')
                # init_weights(self.score_aux2_seg, 'normal')
                # init_weights(self.score_aux1_seg, 'normal')

            init_weights(self.score_aux3, 'normal')
            init_weights(self.score_aux2, 'normal')
            init_weights(self.score_aux1, 'normal')
            init_weights(self.head, 'normal')

        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        self.up1 = Conc_Up_Residual(dims[4], dims[3], norm=norm)
        self.up2 = Conc_Up_Residual(dims[3], dims[2], norm=norm)
        self.up3 = Conc_Up_Residual(dims[2], dims[1], norm=norm)
        self.up4 = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

        self.up_depth = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        self.up_seg = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target_1=None, target_2=None, label=None, out_keys=None, phase='train',
                content_layers=None,
                return_losses=True):
        result = {}
        layer_0 = self.relu(self.bn1(self.conv1(source)))
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.trans:
            # content model branch

            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3)
            result['gen_depth'] = self.up_depth(up4)
            result['gen_seg'] = self.up_seg(up4)

        # segmentation branch
        score_head = self.head(layer_4)

        score_aux1 = None
        score_aux2 = None
        score_aux3 = None
        if self.cfg.WHICH_SCORE == 'main':
            score_aux1 = self.score_aux1(layer_3)
            score_aux2 = self.score_aux2(layer_2)
            score_aux3 = self.score_aux3(layer_1)
        elif self.cfg.WHICH_SCORE == 'up':
            score_aux1 = self.score_aux1(up1)
            score_aux2 = self.score_aux2(up2)
            score_aux3 = self.score_aux3(up3)
        elif self.cfg.WHICH_SCORE == 'both':
            score_aux1 = self.score_aux1(up1 + layer_3)
            score_aux2 = self.score_aux2(up2 + layer_2)
            score_aux3 = self.score_aux3(up3 + layer_1)

        score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux1
        score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux2
        score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux3

        result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_content_depth'] = self.content_model(result['gen_depth'], target_1, layers=content_layers)
            result['loss_content_seg'] = self.content_model(result['gen_seg'], target_2, layers=content_layers)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_pix2pix_depth'] = self.pix2pix_criterion(result['gen_depth'], target_1)
            result['loss_pix2pix_seg'] = self.pix2pix_criterion(result['gen_seg'], target_2)

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class FCN_Conc_MultiModalTarget_Conc(nn.Module):

    def __init__(self, cfg, device=None):
        super(FCN_Conc_MultiModalTarget_Conc, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.head = _FCNHead(512, num_classes, nn.BatchNorm2d)
        # self.head = nn.Conv2d(512, num_classes, 1)

        if self.trans:
            self.build_upsample_content_layers(dims)

        self.score_aux1 = nn.Sequential(
            nn.Conv2d(dims[3], num_classes, 1)
        )

        self.score_aux2 = nn.Sequential(
            nn.Conv2d(dims[2], num_classes, 1)
        )
        self.score_aux3 = nn.Sequential(
            nn.Conv2d(dims[1], num_classes, 1)
        )

        if pretrained:
            init_weights(self.head, 'normal')

            if self.trans:
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')

            init_weights(self.head, 'normal')
            init_weights(self.score_aux3, 'normal')
            init_weights(self.score_aux2, 'normal')
            init_weights(self.score_aux1, 'normal')

        else:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        self.up1 = Conc_Up_Residual(dims[4], dims[3], norm=norm)
        self.up2 = Conc_Up_Residual(dims[3], dims[2], norm=norm)
        self.up3 = Conc_Up_Residual(dims[2], dims[1], norm=norm)
        self.up4 = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

        self.up_image_content = nn.Sequential(
            nn.Conv2d(64, 6, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}

        layer_0 = self.relu(self.bn1(self.conv1(source)))
        if not self.trans:
            layer_0 = self.maxpool(layer_0)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        if self.trans:
            # content model branch

            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3)

            result['gen_img'] = self.up_image_content(up4)

        # segmentation branch
        score_head = self.head(layer_4)

        score_aux1 = None
        score_aux2 = None
        score_aux3 = None
        if self.cfg.WHICH_SCORE == 'main':
            score_aux1 = self.score_aux1(layer_3)
            score_aux2 = self.score_aux2(layer_2)
            score_aux3 = self.score_aux3(layer_1)
        elif self.cfg.WHICH_SCORE == 'up':
            score_aux1 = self.score_aux1(up1)
            score_aux2 = self.score_aux2(up2)
            score_aux3 = self.score_aux3(up3)
        elif self.cfg.WHICH_SCORE == 'both':
            score_aux1 = self.score_aux1(up1 + layer_3)
            score_aux2 = self.score_aux2(up2 + layer_2)
            score_aux3 = self.score_aux3(up3 + layer_1)

        score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux1
        score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux2
        score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux3

        result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
            result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

        return result


class FCN_Conc_Multiscale(nn.Module):

    def __init__(self, cfg, device=None):
        super(FCN_Conc_Multiscale, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.head = _FCNHead(512, num_classes, nn.BatchNorm2d)

        self.score_aux1 = nn.Conv2d(256, num_classes, 1)
        self.score_aux2 = nn.Conv2d(128, num_classes, 1)
        self.score_aux3 = nn.Conv2d(64, num_classes, 1)

        if self.trans:
            self.build_upsample_content_layers(dims, num_classes)

        if pretrained:
            if self.trans:
                # init_weights(self.up_image_14, 'normal')
                init_weights(self.up_image_28, 'normal')
                init_weights(self.up_image_56, 'normal')
                init_weights(self.up_image_112, 'normal')
                init_weights(self.up_image_224, 'normal')
                init_weights(self.up1, 'normal')
                init_weights(self.up2, 'normal')
                init_weights(self.up3, 'normal')
                init_weights(self.up4, 'normal')

            init_weights(self.head, 'normal')
            init_weights(self.score_aux3, 'normal')
            init_weights(self.score_aux2, 'normal')
            init_weights(self.score_aux1, 'normal')

        elif not pretrained:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_content_layers(self, dims, num_classes):

        # norm = nn.InstanceNorm2d
        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d

        self.up1 = Conc_Up_Residual(dims[4], dims[3], norm=norm)
        self.up2 = Conc_Up_Residual(dims[3], dims[2], norm=norm)
        self.up3 = Conc_Up_Residual(dims[2], dims[1], norm=norm)
        self.up4 = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)

        self.up_image_28 = nn.Sequential(
            nn.Conv2d(dims[3], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_image_56 = nn.Sequential(
            nn.Conv2d(dims[2], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_image_112 = nn.Sequential(
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_image_224 = nn.Sequential(
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}
        layer_0 = self.relu(self.bn1(self.conv1(source)))
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        # content model branch
        if self.trans:
            scale_times = self.cfg.MULTI_SCALE_NUM
            ms_compare = []

            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3)

            compare_28 = self.up_image_28(up1)
            compare_56 = self.up_image_56(up2)
            compare_112 = self.up_image_112(up3)
            compare_224 = self.up_image_224(up4)

            ms_compare.append(compare_224)
            ms_compare.append(compare_112)
            ms_compare.append(compare_56)
            ms_compare.append(compare_28)
            # ms_compare.append(compare_14)
            # ms_compare.append(compare_7)

            result['gen_img'] = ms_compare[:scale_times]

        # segmentation branch
        score_head = self.head(layer_4)

        score_aux1 = None
        score_aux2 = None
        score_aux3 = None
        if self.cfg.WHICH_SCORE == 'main':
            score_aux1 = self.score_aux1(layer_3)
            score_aux2 = self.score_aux2(layer_2)
            score_aux3 = self.score_aux3(layer_1)
        elif self.cfg.WHICH_SCORE == 'up':
            score_aux1 = self.score_aux1(up1)
            score_aux2 = self.score_aux2(up2)
            score_aux3 = self.score_aux3(up3)
        elif self.cfg.WHICH_SCORE == 'both':
            score_aux1 = self.score_aux1(up1 + layer_3)
            score_aux2 = self.score_aux2(up2 + layer_2)
            score_aux3 = self.score_aux3(up3 + layer_1)

        score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux1
        score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux2
        score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
        score = score + score_aux3

        result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

        if self.trans and phase == 'train':
            scale_times = self.cfg.MULTI_SCALE_NUM
            trans_loss_list = []
            loss_key = None
            for i, (gen, _target) in enumerate(zip(result['gen_img'], target)):
                assert (gen.size()[-1] == _target.size()[-1])
                # content_layers = [str(layer) for layer in range(5 - i)]
                if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                    loss_key = 'loss_content'
                    trans_loss_list.append(self.content_model(gen, _target, layers=content_layers))

                elif 'PIX2PIX' in self.cfg.LOSS_TYPES:
                    loss_key = 'loss_pix2pix'
                    trans_loss_list.append(self.pix2pix_criterion(gen, _target))

            loss_coef = [1] * scale_times
            ms_losses = [loss_coef[i] * loss for i, loss in enumerate(trans_loss_list)]
            result[loss_key] = sum(ms_losses)

        if 'CLS' in self.cfg.LOSS_TYPES:
            result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, padding=0, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


#######################################################################
class UNet(nn.Module):
    def __init__(self, cfg, device=None):
        super(UNet, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        encoder = cfg.ARCH
        num_classes = cfg.NUM_CLASSES

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        self.score = nn.Conv2d(dims[1], num_classes, 1)

        # norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        self.up1 = Conc_Up_Residual(dims[4], dims[3], norm=nn.BatchNorm2d)
        self.up2 = Conc_Up_Residual(dims[3], dims[2], norm=nn.BatchNorm2d)
        self.up3 = Conc_Up_Residual(dims[2], dims[1], norm=nn.BatchNorm2d)
        self.up4 = Conc_Up_Residual(dims[1], dims[1], norm=nn.BatchNorm2d, conc_feat=False)

        if pretrained:
            init_weights(self.up1, 'normal')
            init_weights(self.up2, 'normal')
            init_weights(self.up3, 'normal')
            init_weights(self.up4, 'normal')
            init_weights(self.score, 'normal')

        else:

            init_weights(self, 'normal')

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):
        result = {}

        layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        up1 = self.up1(layer_4, layer_3)
        up2 = self.up2(up1, layer_2)
        up3 = self.up3(up2, layer_1)
        up4 = self.up4(up3)

        result['cls'] = self.score(up4)
        result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


# class UNet_Long(nn.Module):
#     def __init__(self, cfg, device=None):
#         super(UNet_Long, self).__init__()
#
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#         encoder = cfg.ARCH
#         num_classes = cfg.NUM_CLASSES
#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         if cfg.PRETRAINED == 'place':
#             resnet = models.__dict__['resnet18'](num_classes=365)
#             load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
#             checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
#             state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
#             resnet.load_state_dict(state_dict)
#             print('place resnet18 loaded....')
#         else:
#             resnet = resnet18(pretrained=pretrained)
#             print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#
#         self.score = nn.Conv2d(dims[1], num_classes, 1)
#
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#
#         self.up1 = Conc_Up_Residual(dims[4], dims[3], norm=norm)
#         self.up2 = Conc_Up_Residual(dims[3], dims[2], norm=norm)
#         self.up3 = Conc_Up_Residual(dims[2], dims[1], norm=norm)
#         self.up4 = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)
#
#         if self.trans:
#             self.up_image_content = nn.Sequential(
#                 conv_norm_relu(64, 64, norm=norm),
#                 nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#                 nn.Tanh()
#             )
#
#         if pretrained:
#             init_weights(self.up1, 'normal')
#             init_weights(self.up2, 'normal')
#             init_weights(self.up3, 'normal')
#             init_weights(self.up4, 'normal')
#             init_weights(self.score, 'normal')
#
#             if self.trans:
#                 init_weights(self.up_image_content, 'normal')
#         else:
#
#             init_weights(self, 'normal')
#
#     def set_content_model(self, content_model):
#         self.content_model = content_model
#
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
#                 return_losses=True):
#         result = {}
#
#         layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#
#         up1 = self.up1(layer_4, layer_3)
#         up2 = self.up2(up1, layer_2)
#         up3 = self.up3(up2, layer_1)
#         up4 = self.up4(up3)
#
#         if self.trans:
#             result['gen_img'] = self.up_image_content(up4)
#
#         result['cls'] = self.score(up4)
#
#         if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)
#
#         if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)
#
#         if 'CLS' in self.cfg.LOSS_TYPES:
#             result['loss_cls'] = self.cls_criterion(result['cls'], label)
#
#         return result
#
#
# class UNet_Share_256(nn.Module):
#     def __init__(self, cfg, device=None):
#         super(UNet_Share_256, self).__init__()
#
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#         encoder = cfg.ARCH
#         num_classes = cfg.NUM_CLASSES
#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         if cfg.PRETRAINED == 'place':
#             resnet = models.__dict__['resnet18'](num_classes=365)
#             load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
#             checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
#             state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
#             resnet.load_state_dict(state_dict)
#             print('place resnet18 loaded....')
#         else:
#             resnet = resnet18(pretrained=pretrained)
#             print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#
#         self.score = nn.Conv2d(dims[1], num_classes, 1)
#
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#
#         self.up1 = Conc_Up_Residual(dims[4], dims[3])
#         self.up2 = Conc_Up_Residual(dims[3], dims[2])
#         self.up3 = Conc_Up_Residual(dims[2], dims[1])
#         self.up4 = Conc_Up_Residual(dims[1], dims[1], conc_feat=False)
#
#         if self.trans:
#             self.up2_content = Conc_Up_Residual(dims[3], dims[2])
#             self.up3_content = Conc_Up_Residual(dims[2], dims[1], norm=norm)
#             self.up4_content = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)
#             self.up_image = nn.Sequential(
#                 nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#                 nn.Tanh()
#             )
#
#         if pretrained:
#             init_weights(self.up1, 'normal')
#             init_weights(self.up2, 'normal')
#             init_weights(self.up3, 'normal')
#             init_weights(self.up4, 'normal')
#             init_weights(self.score, 'normal')
#
#             if self.trans:
#                 init_weights(self.up2_content, 'normal')
#                 init_weights(self.up3_content, 'normal')
#                 init_weights(self.up4_content, 'normal')
#                 init_weights(self.up_image, 'normal')
#         else:
#
#             init_weights(self, 'normal')
#
#     def set_content_model(self, content_model):
#         self.content_model = content_model
#
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
#                 return_losses=True):
#         result = {}
#
#         layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#
#         up1 = self.up1(layer_4, layer_3)
#         up2 = self.up2(up1, layer_2)
#         up3 = self.up3(up2, layer_1)
#         up4 = self.up4(up3)
#
#         if self.trans:
#             up2_content = self.up2_content(up1, layer_2)
#             up3_content = self.up3_content(up2_content, layer_1)
#             up4_content = self.up4_content(up3_content)
#             result['gen_img'] = self.up_image(up4_content)
#
#         result['cls'] = self.score(up4)
#
#         if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)
#
#         if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)
#
#         if 'CLS' in self.cfg.LOSS_TYPES:
#             result['loss_cls'] = self.cls_criterion(result['cls'], label)
#
#         return result
#
#
# class UNet_Share_128(nn.Module):
#     def __init__(self, cfg, device=None):
#         super(UNet_Share_128, self).__init__()
#
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#         encoder = cfg.ARCH
#         num_classes = cfg.NUM_CLASSES
#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         if cfg.PRETRAINED == 'place':
#             resnet = models.__dict__['resnet18'](num_classes=365)
#             load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
#             checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
#             state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
#             resnet.load_state_dict(state_dict)
#             print('place resnet18 loaded....')
#         else:
#             resnet = resnet18(pretrained=pretrained)
#             print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#
#         self.score = nn.Conv2d(dims[1], num_classes, 1)
#
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#
#         self.up1 = Conc_Up_Residual(dims[4], dims[3])
#         self.up2 = Conc_Up_Residual(dims[3], dims[2])
#         self.up3 = Conc_Up_Residual(dims[2], dims[1])
#         self.up4 = Conc_Up_Residual(dims[1], dims[1], conc_feat=False)
#
#         if self.trans:
#             self.up3_content = Conc_Up_Residual(dims[2], dims[1], norm=norm)
#             self.up4_content = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)
#             self.up_image = nn.Sequential(
#                 nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#                 nn.Tanh()
#             )
#
#         if pretrained:
#             init_weights(self.up1, 'normal')
#             init_weights(self.up2, 'normal')
#             init_weights(self.up3, 'normal')
#             init_weights(self.up4, 'normal')
#             init_weights(self.score, 'normal')
#
#             if self.trans:
#                 init_weights(self.up3_content, 'normal')
#                 init_weights(self.up4_content, 'normal')
#                 init_weights(self.up_image, 'normal')
#         else:
#
#             init_weights(self, 'normal')
#
#     def set_content_model(self, content_model):
#         self.content_model = content_model
#
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
#                 return_losses=True):
#         result = {}
#
#         layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#
#         up1 = self.up1(layer_4, layer_3)
#         up2 = self.up2(up1, layer_2)
#         up3 = self.up3(up2, layer_1)
#         up4 = self.up4(up3)
#
#         if self.trans:
#             # content
#             up3_content = self.up3_content(up2, layer_1)
#             up4_content = self.up4_content(up3_content)
#             result['gen_img'] = self.up_image(up4_content)
#
#         result['cls'] = self.score(up4)
#
#         if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)
#
#         if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)
#
#         if 'CLS' in self.cfg.LOSS_TYPES:
#             result['loss_cls'] = self.cls_criterion(result['cls'], label)
#
#         return result
#
#
# class UNet_Share_64(nn.Module):
#     def __init__(self, cfg, device=None):
#         super(UNet_Share_64, self).__init__()
#
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#         encoder = cfg.ARCH
#         num_classes = cfg.NUM_CLASSES
#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         if cfg.PRETRAINED == 'place':
#             resnet = models.__dict__['resnet18'](num_classes=365)
#             load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
#             checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
#             state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
#             resnet.load_state_dict(state_dict)
#             print('place resnet18 loaded....')
#         else:
#             resnet = resnet18(pretrained=pretrained)
#             print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#
#         self.score = nn.Conv2d(dims[1], num_classes, 1)
#
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#
#         self.up1 = Conc_Up_Residual(dims[4], dims[3])
#         self.up2 = Conc_Up_Residual(dims[3], dims[2])
#         self.up3 = Conc_Up_Residual(dims[2], dims[1])
#         self.up4 = Conc_Up_Residual(dims[1], dims[1], conc_feat=False)
#
#         if self.trans:
#             self.up4_content = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)
#             self.up_image = nn.Sequential(
#                 nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#                 nn.Tanh()
#             )
#
#         if pretrained:
#             init_weights(self.up1, 'normal')
#             init_weights(self.up2, 'normal')
#             init_weights(self.up3, 'normal')
#             init_weights(self.up4, 'normal')
#             init_weights(self.score, 'normal')
#
#             if self.trans:
#                 init_weights(self.up4_content, 'normal')
#                 init_weights(self.up_image, 'normal')
#         else:
#
#             init_weights(self, 'normal')
#
#     def set_content_model(self, content_model):
#         self.content_model = content_model
#
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
#                 return_losses=True):
#         result = {}
#         layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#
#         up1 = self.up1(layer_4, layer_3)
#         up2 = self.up2(up1, layer_2)
#         up3 = self.up3(up2, layer_1)
#         up4 = self.up4(up3)
#
#         if self.trans:
#             # content
#             up4_content = self.up4_content(up3)
#             result['gen_img'] = self.up_image(up4_content)
#
#         result['cls'] = self.score(up4)
#
#         if 'SEMANTIC' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)
#
#         if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
#             result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)
#
#         if 'CLS' in self.cfg.LOSS_TYPES:
#             result['loss_cls'] = self.cls_criterion(result['cls'], label)
#
#         return result


# class Trans2_UNet_Multiscale(nn.Module):
#
#     def __init__(self, cfg, num_classes=37, encoder='resnet18', using_semantic_branch=True, device=None):
#         super(Trans2_UNet_Multiscale, self).__init__()
#
#         self.encoder = encoder
#         self.cfg = cfg
#         self.using_semantic_branch = using_semantic_branch
#         self.device = device#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         if cfg.PRETRAINED == 'place':
#             resnet = models.__dict__['resnet18'](num_classes=365)
#             load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
#             checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
#             state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
#             resnet.load_state_dict(state_dict)
#             print('place resnet18 loaded....')
#         else:
#             resnet = resnet18(pretrained=pretrained)
#             print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#         self.cls = nn.Conv2d(dims[1], num_classes, kernel_size=1)
#         self.build_upsample_layers(dims)
#
#         # self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
#         # self.fc = nn.Linear(fc_input_nc, cfg.NUM_CLASSES)
#
#         if pretrained:
#
#             init_weights(self.up1, 'normal')
#             init_weights(self.up2, 'normal')
#             init_weights(self.up3, 'normal')
#             init_weights(self.up4, 'normal')
#             init_weights(self.cls, 'normal')
#             init_weights(self.up_image_28, 'normal')
#             init_weights(self.up_image_56, 'normal')
#             init_weights(self.up_image_112, 'normal')
#             init_weights(self.up_image_224, 'normal')
#         else:
#
#             init_weights(self, 'normal')
#
#     def set_content_model(self, content_model):
#         self.content_model = content_model
#
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def build_upsample_layers(self, dims):
#
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#
#         self.up1 = UpConv_Conc(dims[4], dims[3], norm=norm)
#         self.up2 = UpConv_Conc(dims[3], dims[2], norm=norm)
#         self.up3 = UpConv_Conc(dims[2], dims[1], norm=norm)
#         self.up4 = UpConv_Conc(dims[1], dims[1], norm=norm, if_conc=False)
#
#         self.up_image_28 = nn.Sequential(
#             nn.Conv2d(dims[3], 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#         self.up_image_56 = nn.Sequential(
#             nn.Conv2d(dims[2], 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#         self.up_image_112 = nn.Sequential(
#             nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#         self.up_image_224 = nn.Sequential(
#             nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#
#     def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
#                 return_losses=True):
#         result = {}
#
#         layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#
#         up1 = self.up1(layer_4, layer_3)
#         up2 = self.up2(up1, layer_2)
#         up3 = self.up3(up2, layer_1)
#         up4 = self.up4(up3)
#
#         scale_times = self.cfg.MULTI_SCALE_NUM
#         ms_compare = []
#
#         compare_28 = self.up_image_28(up1)
#         compare_56 = self.up_image_56(up2)
#         compare_112 = self.up_image_112(up3)
#         compare_224 = self.up_image_224(up4)
#
#         ms_compare.append(compare_224)
#         ms_compare.append(compare_112)
#         ms_compare.append(compare_56)
#         ms_compare.append(compare_28)
#         # ms_compare.append(compare_14)
#         # ms_compare.append(compare_7)
#
#         result['gen_img'] = ms_compare[:scale_times]
#         result['cls'] = self.cls(up4)
#
#         if self.using_semantic_branch and phase == 'train':
#             scale_times = self.cfg.MULTI_SCALE_NUM
#             content_loss_list = []
#             for i, (gen, _target) in enumerate(zip(result['gen_img'], target)):
#                 assert (gen.size()[-1] == _target.size()[-1])
#                 # content_layers = [str(layer) for layer in range(5 - i)]
#                 content_loss_list.append(self.content_model(gen, _target, layers=content_layers))
#
#             loss_coef = [1] * scale_times
#             ms_losses = [loss_coef[i] * loss for i, loss in enumerate(content_loss_list)]
#             result['loss_content'] = sum(ms_losses)
#
#         if 'CLS' in self.cfg.LOSS_TYPES:
#             result['loss_cls'] = self.cls_criterion(result['cls'], label)
#
#         return result


# class Trans2_UNet(nn.Module):
#
#     def __init__(self, cfg, num_classes=37, encoder='resnet18', using_semantic_branch=True, device=None):
#         super(Trans2_UNet, self).__init__()
#
#         self.encoder = encoder
#         self.cfg = cfg
#         self.using_semantic_branch = using_semantic_branch
#         self.device = device
#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         if cfg.PRETRAINED == 'place':
#             resnet = models.__dict__['resnet18'](num_classes=365)
#             load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
#             checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
#             state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
#             resnet.load_state_dict(state_dict)
#             print('place resnet18 loaded....')
#         else:
#             resnet = resnet18(pretrained=pretrained)
#             print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#         self.cls = nn.Conv2d(dims[1], num_classes, kernel_size=1)
#
#         self.build_upsample_layers(dims, num_classes)
#
#         # self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
#         # self.fc = nn.Linear(fc_input_nc, cfg.NUM_CLASSES)
#
#         if pretrained:
#
#             init_weights(self.up1, 'normal')
#             init_weights(self.up2, 'normal')
#             init_weights(self.up3, 'normal')
#             init_weights(self.up4, 'normal')
#             init_weights(self.cls, 'normal')
#             if using_semantic_branch:
#
#                 init_weights(self.up_image, 'normal')
#
#         else:
#
#             init_weights(self, 'normal')
#
#     def set_content_model(self, content_model):
#         self.content_model = content_model
#
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def build_upsample_layers(self, dims, num_classes):
#
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#
#         self.up1 = UpsampleBasicBlock(dims[4], dims[3], norm=norm)
#         self.up2 = UpsampleBasicBlock(dims[3] * 2, dims[2], norm=norm)
#         self.up3 = UpsampleBasicBlock(dims[2] * 2, dims[1], norm=norm)
#         self.up4 = UpsampleBasicBlock(dims[1] * 2, dims[1], norm=norm)
#         self.up_image = nn.Sequential(
#             conv_norm_relu(dims[1], dims[1], norm=norm),
#             nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#
#     def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
#                 return_losses=True):
#         result = {}
#
#         layer_0 = self.relu(self.bn1(self.conv1(source)))
#         layer_1 = self.layer1(layer_0)
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#
#         up1 = self.up1(layer_4)
#         up2 = self.up2(torch.cat((up1, layer_3), 1))
#         up3 = self.up3(torch.cat((up2, layer_2), 1))
#         up4 = self.up4(torch.cat((up3, layer_1), 1))
#
#         if self.using_semantic_branch:
#             result['gen_img'] = self.up_image(up4)
#
#         result['cls'] = self.cls(up4)
#
#         if self.using_semantic_branch and phase == 'train':
#             result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)
#
#         if 'CLS' in self.cfg.LOSS_TYPES:
#             result['loss_cls'] = self.cls_criterion(result['cls'], label)
#
#         return result

########### PSPNET ###############
# class PPM(nn.Module):
#     def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
#         super(PPM, self).__init__()
#         self.features = []
#         for bin in bins:
#             self.features.append(nn.Sequential(
#                 nn.AdaptiveAvgPool2d(bin),
#                 nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
#                 BatchNorm(reduction_dim),
#                 nn.ReLU(inplace=True)
#             ))
#         self.features = nn.ModuleList(self.features)
#
#     def forward(self, x):
#         x_size = x.size()
#         out = [x]
#         for f in self.features:
#             out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
#         return torch.cat(out, 1)


# class PSP_Conc(nn.Module):
#
#     def __init__(self, cfg, device=None, SyncBatchNorm=None, arch='resnet50', bins=(1, 2, 3, 6), dropout=0.1, classes=19,
#                  zoom_factor=8, use_ppm=True, pretrained=True):
#         super(PSP_Conc, self).__init__()
#         # assert layers in [50, 101, 152]
#         self.cfg = cfg
#         assert 2048 % len(bins) == 0
#         assert classes > 1
#         assert zoom_factor in [1, 2, 4, 8]
#         self.zoom_factor = zoom_factor
#         self.device = device
#         self.use_ppm = use_ppm
#         self.trans = not cfg.NO_TRANS
#         # models.BatchNorm = SyncBatchNorm
#         # BatchNorm = SyncBatchNorm
#         self.norm = SyncBatchNorm
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         resnet = models.__dict__[arch](pretrained=pretrained, )
#         print("load ", arch)
#
#         # self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
#         self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
#                                     resnet.conv3, resnet.bn3, resnet.relu)
#         self.maxpool = resnet.maxpool
#         self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
#
#         for n, m in self.layer3.named_modules():
#             if 'conv2' in n:
#                 m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
#             elif 'downsample.0' in n:
#                 m.stride = (1, 1)
#         for n, m in self.layer4.named_modules():
#             if 'conv2' in n:
#                 m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
#             elif 'downsample.0' in n:
#                 m.stride = (1, 1)
#         if self.trans:
#             self.build_upsample_content_layers(dims)
#
#         fea_dim = 2048
#         if use_ppm:
#             self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins, self.norm)
#             fea_dim *= 2
#         self.cls = nn.Sequential(
#             nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
#             self.norm(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=dropout),
#             nn.Conv2d(512, classes, kernel_size=1)
#         )
#         # if self.training:
#         self.aux = nn.Sequential(
#             nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
#             self.norm(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=dropout),
#             nn.Conv2d(256, classes, kernel_size=1)
#         )
#
#         if self.trans:
#
#             init_weights(self.up1, 'normal')
#             init_weights(self.up2, 'normal')
#             init_weights(self.up3, 'normal')
#             init_weights(self.up4, 'normal')
#             init_weights(self.up_seg, 'normal')
#         init_weights(self.aux, 'normal')
#         init_weights(self.cls, 'normal')
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def set_content_model(self, content_model):
#         self.content_model = content_model
#
#     def build_upsample_content_layers(self, dims):
#
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#         # norm = self.norm
#         self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm, upsample=False)
#         self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
#         self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
#         self.up4 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)
#
#         self.up_seg = nn.Sequential(
#             nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#
#     def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
#                 return_losses=True):
#         result = {}
#         x = source
#         y = label
#         layer_0 = self.layer0(x)
#         if not self.trans:
#             layer_0 = self.maxpool(layer_0)
#         layer_1 = self.layer1(layer_0)
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#         # print(x.size())
#         print(layer_3.size())
#         print(layer_4.size())
#
#         if self.use_ppm:
#             x = self.ppm(layer_4)
#
#         if phase == 'train':
#             aux = self.aux(layer_3)
#             aux = F.interpolate(aux, source.size()[2:], mode='bilinear', align_corners=True)
#             main_loss = self.cls_criterion(result['cls'], y)
#             aux_loss = self.cls_criterion(aux, y)
#             result['loss_cls'] = main_loss + 0.4 * aux_loss
#         else:
#             main_loss = self.cls_criterion(result['cls'], y)
#             result['loss_cls'] = main_loss
#         if self.trans and phase == 'train':
#             up1_seg = self.up1(layer_4, layer_3)
#             up2_seg = self.up2(up1_seg, layer_2)
#             up3_seg = self.up3(up2_seg, layer_1)
#             up4_seg = self.up4(up3_seg)
#             result['gen_img'] = self.up_seg(up4_seg)
#             print(result['gen_img'].size())
#
#             result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)
#
#         x = self.cls(x)
#         result['cls'] = F.interpolate(x, source.size()[2:], mode='bilinear', align_corners=True)
#         return result


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):

    def __init__(self, cfg, BatchNorm=None, bins=(1, 2, 3, 6), dropout=0.1,
                 zoom_factor=8, use_ppm=True, pretrained=True, device=None):
        super(PSPNet, self).__init__()
        assert 2048 % len(bins) == 0
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        models.BatchNorm = BatchNorm
        self.device = device
        self.trans = not cfg.NO_TRANS
        self.cfg = cfg
        dims = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        if self.trans:
            self.build_upsample_content_layers(dims)

        resnet = models.__dict__[cfg.ARCH](pretrained=pretrained, deep_base=True)
        print("load ", cfg.ARCH)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins, BatchNorm)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, cfg.NUM_CLASSES, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, cfg.NUM_CLASSES, kernel_size=1)
            )

        if self.trans:
            init_weights(self.up1, 'normal')
            init_weights(self.up2, 'normal')
            init_weights(self.up3, 'normal')
            init_weights(self.up4, 'normal')
            init_weights(self.cross_1, 'normal')
            init_weights(self.cross_2, 'normal')
            init_weights(self.cross_3, 'normal')
            init_weights(self.up_seg, 'normal')

        init_weights(self.aux, 'normal')
        init_weights(self.cls, 'normal')
        init_weights(self.ppm, 'normal')

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        # norm = self.norm
        self.cross_1 = nn.Conv2d(dims[7], dims[4], kernel_size=1, bias=False)
        self.cross_2 = nn.Conv2d(dims[6], dims[3], kernel_size=1, bias=False)
        self.cross_3 = nn.Conv2d(dims[5], dims[3], kernel_size=1, bias=False)
        self.up1 = Conc_Residual_bottleneck(dims[5], dims[4], norm=norm)
        self.up2 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
        self.up3 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
        self.up4 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)

        self.up_seg = nn.Sequential(
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        self.score_head = nn.Conv2d(512, self.cfg.NUM_CLASSES, 1)
        self.score_aux1 = nn.Conv2d(256, self.cfg.NUM_CLASSES, 1)
        self.score_aux2 = nn.Conv2d(128, self.cfg.NUM_CLASSES, 1)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def set_content_model(self, content_model):
        self.content_model = content_model.to(self.device)

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None,
                return_losses=True):

        x = source
        y = label
        result = {}
        # x_size = x.size()
        # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        # h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        # w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        layer_0 = self.layer0(x)
        layer_1 = self.layer1(self.maxpool(layer_0))
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)
        # print('x', x.size())
        # print('layer_0', layer_0.size())
        # print('layer_1', layer_1.size())
        # print('layer_2', layer_2.size())
        # print('layer_3', layer_3.size())
        # print('layer_4', layer_4.size())

        # print(x.size())

        x = layer_4
        if self.use_ppm:
            x = self.ppm(x)

        if not self.trans:
            x = self.cls(x)
            if self.zoom_factor != 1:
                result['cls'] = F.interpolate(x, size=source.size()[2:], mode='bilinear', align_corners=True)
        else:
            cross_1 = self.cross_1(x)
            cross_2 = self.cross_2(layer_4)
            cross_3 = self.cross_3(layer_3)
            cross_conc = torch.cat((cross_1, cross_2, cross_3), 1)
            up1_seg = self.up1(cross_conc, layer_2)
            up2_seg = self.up2(up1_seg, layer_1)
            up3_seg = self.up3(up2_seg, layer_0)
            up4_seg = self.up4(up3_seg)

            result['gen_img'] = self.up_seg(up4_seg)
            result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)

            score_head = self.score_head(up1_seg)
            score_aux1 = self.score_aux1(up2_seg)
            score_aux2 = self.score_aux2(up3_seg)

            x = self.cls(x)
            score = F.interpolate(x, score_head.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_head
            score = F.interpolate(score, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux1
            score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux2
            result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

        if return_losses:

            if phase == 'train' and not self.trans:

                aux = self.aux(layer_3)
                if self.zoom_factor != 1:
                    aux = F.interpolate(aux, size=source.size()[2:], mode='bilinear', align_corners=True)
                main_loss = self.cls_criterion(result['cls'], y)
                aux_loss = self.cls_criterion(aux, y)
                # print("x:",result['cls'].size())
                # print("y:",y.size())
                # print("main_loss:",main_loss.size())
                result['loss_cls'] = main_loss + 0.4 * aux_loss

            else:
                main_loss = self.cls_criterion(result['cls'], y)
                result['loss_cls'] = main_loss

        return result
