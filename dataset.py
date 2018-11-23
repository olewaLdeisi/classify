# -*- coding:utf-8 -*-
import os
import random
import logging
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader
from PIL import Image
# from compiler.ast import flatten  # python2
import torchvision.transforms.functional as F_trans
import operator # python3
from functools import reduce # python3

classes = ['Normal','Glaucoma']
# classes = ['Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau', 'Baroque',
#     'Color_Field_Painting', 'Contemporary_Realism', 'Cubism', 'Early_Renaissance', 'Expressionism', 'Fauvism',
#     'High_Renaissance', 'Impressionism', 'Mannerism_Late_Renaissance', 'Minimalism', 'Naive_Art_Primitivism',
#     'New_Realism', 'Northern_Renaissance', 'Pointillism', 'Pop_Art', 'Post_Impressionism', 'Realism', 'Rococo',
#     'Romanticism', 'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e', ]


class Dataset(data.Dataset):
    def __init__(self, class_num, data_path, file_path, grayscale=False, transform=None,p=0.0):
        '''
        :param class_num: 类的数量，默认为27（int or list）
        :param data_path: txt路径及文件名 （str）
        :param file_path: 数据集根目录 （str）
        :param grayscale: 是否打开为灰度图，默认为False （bool）
        :param transform: 预处理函数，默认为None
        :param p: 水平翻转概率
        '''
        self.data_path = data_path
        self.file_path = file_path
        self.grayscale = grayscale
        self.transform = transform
        self.p = p
        self.img = [[] for i in range(len(classes))]
        self.label = [[] for i in range(len(classes))]
        # 读取.txt文件数据
        with open(self.data_path, 'r') as f:
            for line in f:
                img, label = line.strip().split('@')
                self.img[eval(label)].append(img)
                self.label[eval(label)].append(label)
        if isinstance(class_num, int):
            # python3
            # self.img = reduce(operator.add, self.img[:class_num])
            # self.label = reduce(operator.add, self.label[:class_num])
            # python2
            self.bincount = [len(i) for i in self.label]  # 各个类的样本数
            # rand_index_list = [random.sample(range(self.bincount[i]), 20) for i in range(class_num)]
            # self.img = [self.img[i][j] for i in range(class_num) for j in rand_index_list[i]]
            # self.label = [self.label[i][j] for i in range(class_num) for j in rand_index_list[i]]
            self.img = reduce(operator.add, self.img[:class_num])
            self.label = reduce(operator.add, self.label[:class_num])
            self.labellist = [i for i in range(class_num)]
        elif isinstance(class_num, list):
            # python3
            # self.img = reduce(operator.add, [self.img[i] for i in class_num])
            # self.label = reduce(operator.add, [self.label[i] for i in class_num])
            # python2
            self.bincount = [len(self.label[i]) for i in class_num]
            # rand_index_list = [random.sample(range(self.bincount[i]), 20) for i in range(len(class_num))]
            # self.img = [self.img[i][j] for index, i in enumerate(class_num) for j in rand_index_list[index]]
            # self.label = [self.label[i][j] for index, i in enumerate(class_num) for j in rand_index_list[index]]
            self.img = reduce(operator.add, [self.img[i] for i in class_num])
            self.label = reduce(operator.add, [self.label[i] for i in class_num])
            self.labellist = [i for i in class_num]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img_path = os.path.join(self.file_path, 'img/' + self.img[index])
        mark1_path = os.path.join(self.file_path, 'mark1/' + self.img[index])
        mark2_path = os.path.join(self.file_path, 'mark2/' + self.img[index])
        # 是否打开为灰度图
        if self.grayscale:
            img = Image.open(img_path).convert('L')
        else:
            img = Image.open(img_path).convert('RGB')
        # 视杯视盘语义分割
        mark1 = Image.open(mark1_path).convert('L')
        mark2 = Image.open(mark2_path).convert('L')

        # 0.5 概率水平翻转
        if self.p > 0:
            if random.random() < self.p:
                img = F_trans.hflip(img)
                mark1 = F_trans.hflip(mark1)
                mark2 = F_trans.hflip(mark2)
        # 数据预处理
        if self.transform:
            img = self.transform(img)
            mark1 = self.transform(mark1)
            mark2 = self.transform(mark2)
            img = transforms.Normalize((0.3938,  0.3938,  0.2248), ( 0.1569,  0.1470,  0.1451))(img)
            mark1 = transforms.Normalize((0.1686,), (0.2723,))(mark1)
            mark2 = transforms.Normalize((1.00000e-02 * 8.3514,), (0.1773,))(mark2)
            mark = [mark1,mark2]

        label = eval(self.label[index])
        # 返回图片，对应标签
        # print(self.img[index],self.labellist.index(label))
        return img, mark , self.labellist.index(label)


if __name__ == '__main__':
    logging.basicConfig(filename='loss.log', level=logging.INFO, filemode='a',
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M', )
    # 一些配置参数
    parser = argparse.ArgumentParser(description='Train for Classify')
    parser.add_argument('--model_name', type=str, default='resnet50', help='模型名')
    parser.add_argument('--use_gpu', type=int, default=1, help='是否使用gpu加速')
    parser.add_argument('--crop', type=int, default=1, help='是否中心切割')
    parser.add_argument('--class_num', type=str, default='27', help='分类数量')
    parser.add_argument('--input_height', type=int, default=256, help='切割后图片高度')
    parser.add_argument('--input_weight', type=int, default=256, help='切割后图片宽度')
    parser.add_argument('--output_height', type=int, default=256, help='Resize后图片高度')
    parser.add_argument('--output_weight', type=int, default=256, help='Resize后图片宽度')
    parser.add_argument('--data_path', type=str, default='./data/', help='加载数据集的txt路径文件名')
    parser.add_argument('--file_path', type=str, default='./data/', help='数据集根目录')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=200, help='epoch总数')
    parser.add_argument('--batch_size', type=int, default=5, help='mini-batch大小')
    parser.add_argument('--momentum', type=float, default=0.9, help='优化器动量因子')
    parser.add_argument('--weight_decay', type=float, default=0, help='优化器权重衰减值')
    parser.add_argument('--step_size', type=int, default=7, help='学习率更新间隔')
    parser.add_argument('--gamma', type=float, default=0.1, help='学习率衰减因子')
    parser.add_argument('--pretrained', type=int, default=0, help='是否使用预训练模型参数')
    parser.add_argument('--batchnormal', type=int, default=1, help='是否加入batchnormal层')
    parser.add_argument('--poolmode', type=str, default='Max', help='池化方式')
    parser.add_argument('--dataset_name', type=str, default='wikiart', help='数据集名字')
    parser.add_argument('--viz', type=int, default=1, help='是否使用visdom可视化')
    parser.add_argument('--device_id', type=int, default=0, help='使用GPU的ID')
    # parser.add_argument('--normal',type=str,default='(0.5,0.5,0.5),(0.5,0.5,0.5)',help='归一化均值与标准差')
    parser.add_argument('--decay_scheduler', type=int, default=1, help='是否衰减学习率')
    args = parser.parse_args()
    logging.info(args)

    if args.crop:
        transform = transforms.Compose([transforms.Resize(args.input_height),
                                        transforms.CenterCrop((args.output_height, args.output_weight)),
                                        transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize(args.output_height),
                                        transforms.ToTensor()])
    # args.class_num = '[1,2,6,10,16,18,24]'
    # 因为图片是 3*32*32的，但是pytorch接受的输入是4维的，所以要添加一个1 的维度，相当于变成了1*3*32*32
    train_set = Dataset(class_num=eval(args.class_num), data_path=os.path.join(args.data_path, 'train.txt'),
                        file_path=args.file_path, grayscale=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    val_set = Dataset(class_num=eval(args.class_num), data_path=os.path.join(args.data_path, 'test.txt'),
                      file_path=args.file_path, grayscale=False, transform=transform)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    dataloader = {'train': train_loader, 'val': val_loader}
    total = len(train_set) + len(val_set)
    img_sum = 0.
    for phase in ['train', 'val']:
        print("=" * 5 + phase+" begin" + "=" * 5)
        for i, (img, mark, label) in enumerate(dataloader[phase]):
            pass
            # img = img.cuda(0)
            # img = img.view(img.size(0), img.size(1), -1)
            # temp = torch.sum(img, 2).squeeze(0)
            # img_sum = temp if phase == 'train' and i == 0 else img_sum + temp
    img_mean = img_sum / (args.output_weight * args.output_height * total)

    for phase in ['train', 'val']:
        for i, (img, mark, label) in enumerate(dataloader[phase]):
            img = img.cuda(0)
            img = img.view(img.size(0), img.size(1), -1)
            temp1 = torch.sum((img[:, 0] - img_mean[0]) ** 2, 1)
            temp2 = torch.sum((img[:, 1] - img_mean[1]) ** 2, 1)
            temp3 = torch.sum((img[:, 2] - img_mean[2]) ** 2, 1)
            temp = torch.cat((temp1, temp2, temp3), 0)
            img_sum = temp if phase == 'train' and i == 0 else img_sum + temp
    img_std = torch.sqrt(img_sum / (args.output_weight * args.output_height * total))
    print(img_mean, img_std)
    # datasets = Dataset(class_num=class_num, data_path=data_path, file_path=file_path, grayscale=False,
    #                    transform=transform)
    # vgg19 = torchvision.models.vgg19(pretrained=False)  # 不带BatchNormal层的vgg
    # vgg19 = torchvision.models.vgg19_bn(pretrained=False)  # 带BatchNormal层的vgg, pretrained=True则下载预训练的
    # vgg19.classifier[0] = torch.nn.Linear(512 * (output_height // 32) * (output_weight // 32), 4096, bias=True)
    # vgg19.classifier[6] = torch.nn.Linear(4096, class_num, bias=True)  # 调整分类数目  # transforms.Normalize里的参数是归一化
    # poolindex = [6,13,26,39,52]
    # for i in poolindex:
    #     vgg19.features[i] = torch.nn.AvgPool2d(2, 2, 0, False, True)
    # print(len(list(vgg19.parameters())))
    # print(vgg19)
    #
    # for para in list(vgg19.parameters()):
    #     print(para)
    # print(111)
    # vgg19 = torchvision.models.vgg19(pretrained=False)
    # print(vgg19.features[0:7])
    # print(vgg19.features[7:14])
    # print(vgg19.features[14:27])
    # print(vgg19.features[27:40])
    # print(vgg19.features[40:53])

    # [6,13,26,39,52]
    # [4,9,18,27,36]
    '''
    print("111")
    for i,(img,_) in enumerate(train_loader):
        img = img.transpose(0,1).contiguous()
        img = img.view(img.size(0), -1)
        mean = torch.mean(img, 1)
        std = torch.std(img, 1)
        break
    print(mean, std)
    '''
