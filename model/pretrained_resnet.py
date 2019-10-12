import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):

    def __init__(self, resnet=None, cfg=None, in_channel=3):
        super(ResNet, self).__init__()

        if resnet == 'resnet18':

            if cfg.CONTENT_PRETRAINED == 'place':
                resnet_model = models.__dict__['resnet18'](num_classes=365)
                load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
                checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
                resnet_model.load_state_dict(state_dict)
                print('content model pretrained using place')
            else:
                resnet_model = models.resnet18(True)
                print('content model pretrained using imagenet')

        elif resnet == 'resnet50':

            resnet_model = models.resnet50(True)

        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4

        if in_channel != 3:
            conv1_filter = self.conv1.weight.data / 2 + 0.5
            mean_tensor = torch.mean(conv1_filter, 1, keepdim=True).repeat(1, in_channel, 1, 1) / 2 + 0.5
            self.conv1.weight.data = mean_tensor
            self.conv1.in_channels = in_channel

    def forward(self, x, out_keys):

        out = {}
        out['0'] = self.relu(self.bn1(self.conv1(x)))
        # out['1'] = self.layer1(out['0'])
        out['1'] = self.layer1(self.maxpool(out['0']))
        out['2'] = self.layer2(out['1'])
        out['3'] = self.layer3(out['2'])
        out['4'] = self.layer4(out['3'])
        return [out[key] for key in out_keys]
