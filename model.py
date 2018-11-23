# -*- coding:utf-8 -*-
import torch
import torchvision
from torch import nn
import math

class myVgg19(nn.Module):
    def __init__(self,args):
        super(myVgg19,self).__init__()
        # 定义卷积层
        if args.batchnormal:
            model = torchvision.models.vgg19_bn(pretrained=False)
            poolindex = [6,6,12,12,12]
            self.conv1 = model.features[0:7]
            self.conv2 = model.features[7:14]
            self.conv3 = model.features[14:27]
            self.conv4 = model.features[27:40]
            self.conv5 = model.features[40:53]
        else:
            model = torchvision.models.vgg19(pretrained=False)
            poolindex = [4,4,8,8,8]
            self.conv1 = model.features[0:5]
            self.conv2 = model.features[5:10]
            self.conv3 = model.features[10:19]
            self.conv4 = model.features[19:28]
            self.conv5 = model.features[28:37]
        # # 修改池化层
        if args.poolmode == 'Average':
            self.conv1[poolindex[0]] = torch.nn.AvgPool2d(2, 2, 0)
            self.conv2[poolindex[1]] = torch.nn.AvgPool2d(2, 2, 0)
            self.conv3[poolindex[2]] = torch.nn.AvgPool2d(2, 2, 0)
            self.conv4[poolindex[3]] = torch.nn.AvgPool2d(2, 2, 0)
            self.conv5[poolindex[4]] = torch.nn.AvgPool2d(2, 2, 0)
        for p in self.parameters():
            p.requires_grad = False
        # 定义全连接层
        self.fc = model.classifier
        self.fc[0] = torch.nn.Linear(512 * (args.output_height // 32) * (args.output_weight // 32), 4096, bias=True)
        self.fc[6] = torch.nn.Linear(4096, args.class_num, bias=True)  # 调整分类数目

        if not args.pretrained:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x = x5.view(x.size(0), -1)
        x = self.fc(x)
        pool_output = [x1,x2,x3,x4,x5]
        return x,pool_output

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train for Vgg19')
    parser.add_argument('--use_gpu', type=int, default=1, help='是否使用gpu加速')
    parser.add_argument('--crop',type=int,default=1,help='是否中心切割')
    parser.add_argument('--class_num',type=int,default=27,help='分类数量')
    parser.add_argument('--input_height',type=int,default=256,help='切割后图片高度')
    parser.add_argument('--input_weight',type=int,default=256,help='切割后图片宽度')
    parser.add_argument('--output_height',type=int,default=256,help='Resize后图片高度')
    parser.add_argument('--output_weight',type=int,default=256,help='Resize后图片宽度')
    parser.add_argument('--data_path',type=str,default='.',help='加载数据集的txt路径文件名')
    parser.add_argument('--file_path',type=str,default='.',help='数据集根目录')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=25, help='epoch总数')
    parser.add_argument('--batch_size', type=int, default=5, help='mini-batch大小')
    parser.add_argument('--momentum', type=float, default=0.9, help='优化器动量因子')
    parser.add_argument('--weight_decay', type=float, default=0, help='优化器权重衰减值')
    parser.add_argument('--step_size', type=int, default=7, help='学习率更新间隔')
    parser.add_argument('--gamma', type=float, default=0.1, help='学习率衰减因子')
    parser.add_argument('--pretrained', type=int, default=1, help='是否使用预训练模型参数')
    parser.add_argument('--batchnormal', type=int, default=1, help='是否加入batchnormal层')
    parser.add_argument('--poolmode', type=str, default='Average', help='池化方式')
    args = parser.parse_args()
    model = myVgg19(args)
    print(model)
