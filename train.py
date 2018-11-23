# -*- coding:utf-8 -*-
import os
import logging
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from resnet import *
from vggnet import *
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from dataset import Dataset
import tqdm
import visdom
import pandas as pd

def train_model(model, dataloader, args, optimizer, scheduler):
    if args.viz:
        model_name = args.model_name + "_inchannel="+ str(args.in_channels)
        if args.batchnormal:
            model_name += '_bn'
        if args.pretrained:
            model_name += '_pred'
        model_name += '_Adam_addclassweight' + '_class='+str(args.class_num)
        viz = visdom.Visdom(env=model_name)

    input_img = torch.Tensor(args.batch_size, args.in_channels, args.output_height, args.output_weight)
    input_label = torch.LongTensor(args.batch_size)

    train_weight = torch.Tensor(args.train_bincount)
    train_weight = args.train_n_samples / (args.class_num * train_weight)
    train_weight = train_weight.cuda(args.device_id)

    val_weight = torch.Tensor(args.val_bincount)
    val_weight = args.val_n_samples / (args.class_num * val_weight)
    val_weight = val_weight.cuda(args.device_id)

    if args.use_classweight:
        criterion_total = {'train':nn.CrossEntropyLoss(weight=train_weight),
                           'val':nn.CrossEntropyLoss(weight=val_weight)}
    else:
        criterion_total = {'train': nn.CrossEntropyLoss(),
                           'val': nn.CrossEntropyLoss()}

    sincetime = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    trainset_sizes = 0
    valset_sizes = 0
    train_switch = True
    val_switch = True
    train_epoch_loss_list = []
    val_epoch_loss_list = []
    train_epoch_acc_list = []
    val_epoch_acc_list = []
    for epoch in tqdm.tqdm(range(args.num_epochs)):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        train_running_loss = 0.0
        train_running_corrects = torch.Tensor([0] * args.class_num)
        train_running_total = torch.Tensor([0] * args.class_num)

        val_running_loss = 0.0
        val_running_corrects = torch.Tensor([0] * args.class_num)
        val_running_total = torch.Tensor([0] * args.class_num)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            criterion = criterion_total[phase]
            if phase == 'train':
               if args.decay_scheduler:
                   scheduler.step()
               model.train(True)
            else:
               model.train(False)

            # Iterate over data.
            for i,(imgs, marks, labels) in enumerate(dataloader[phase]):
                # 合并输入通道
                if args.in_channels > 3:
                    inputs = input_img.copy_(torch.cat((imgs, *marks), 1))
                else:
                    inputs = input_img.copy_(imgs)
                labels = input_label.copy_(labels)

                if phase == 'train' and train_switch:
                    trainset_sizes += inputs.size(0)
                if phase == 'val' and val_switch:
                    valset_sizes += inputs.size(0)
                # wrap them in Variable
                if args.use_gpu and torch.cuda.is_available():
                    inputs = inputs.cuda(args.device_id)
                    labels = labels.cuda(args.device_id)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                    train_running_loss += loss.item()
                    train_running_corrects += torch.Tensor([float(torch.sum((preds == j) & (labels == j)))  for j in range(args.class_num)])
                    train_running_total += torch.Tensor([float(torch.sum(labels == j))  for j in range(args.class_num)])
                    # train_running_corrects += torch.sum(preds == labels)
                else:
                    val_running_loss += loss.item()
                    val_running_corrects += torch.Tensor([float(torch.sum((preds == j) & (labels == j))) for j in range(args.class_num)])
                    val_running_total += torch.Tensor([float(torch.sum(labels == j)) for j in range(args.class_num)])
                    # val_running_corrects += torch.sum(preds == labels)

            if phase == 'train' and train_switch:
                train_switch = False
            if phase == 'val' and val_switch:
                val_switch = False
        train_epoch_loss = train_running_loss / trainset_sizes
        # train_epoch_acc = torch.mean(train_running_corrects / train_running_total)
        train_epoch_acc = float(torch.sum(train_running_corrects)) / float(trainset_sizes)

        val_epoch_loss = val_running_loss / valset_sizes
        # val_epoch_acc = torch.mean(val_running_corrects / val_running_total)
        val_epoch_acc = float(torch.sum(val_running_corrects)) / float(valset_sizes)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    args.data_path, train_epoch_loss, train_epoch_acc))

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    args.data_path, val_epoch_loss, val_epoch_acc))
        # 可视化
        if args.viz :
            train_epoch_loss_list.append(train_epoch_loss)
            val_epoch_loss_list.append(val_epoch_loss)
            train_epoch_acc_list.append(float(train_epoch_acc))
            val_epoch_acc_list.append(float(val_epoch_acc))
            if epoch == 0:
                win = viz.line(X=torch.Tensor([[epoch] * 4]),
                                Y=torch.Tensor([[train_epoch_loss,
                                          val_epoch_loss,
                                          train_epoch_acc,
                                          val_epoch_acc]]),
                                opts=dict(legend=['train_epoch_loss',
                                                  'val_epoch_loss',
                                                  'train_epoch_acc',
                                                  'val_epoch_acc']))
            else:
                viz.line(X=torch.Tensor([[epoch] * 4]),
                         Y=torch.Tensor([[train_epoch_loss,
                                          val_epoch_loss,
                                          train_epoch_acc,
                                          val_epoch_acc]]),
                         opts=dict(legend=['train_epoch_loss',
                                           'val_epoch_loss',
                                           'train_epoch_acc',
                                           'val_epoch_acc']),
                         win=win,
                         update='append')
        # deep copy the model
        if phase == 'val' and val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = model.state_dict()

    time_elapsed = time.time() - sincetime
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    dataframe = pd.DataFrame({'train_epoch_loss':train_epoch_loss_list,
                              'val_epoch_loss':val_epoch_loss_list,
                              'train_epoch_acc':train_epoch_acc_list,
                              'val_epoch_acc':val_epoch_acc_list
                              },index=[i for i in range(len(train_epoch_loss_list))])
    dataframe.to_csv(args.model_name+'_class='+str(args.class_num) + '.csv')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    logging.basicConfig(filename='loss.log', level=logging.INFO, filemode='a',
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M', )
    # 一些配置参数
    parser = argparse.ArgumentParser(description='Train for Classify')
    parser.add_argument('--model_name',type=str,default='resnet50',help='模型名')
    parser.add_argument('--use_gpu', type=int, default=1, help='是否使用gpu加速')
    parser.add_argument('--crop',type=int,default=1,help='是否中心切割')
    parser.add_argument('--class_num',type=str,default='27',help='分类数量')
    parser.add_argument('--input_height',type=int,default=256,help='切割后图片高度')
    parser.add_argument('--input_weight',type=int,default=256,help='切割后图片宽度')
    parser.add_argument('--output_height',type=int,default=256,help='Resize后图片高度')
    parser.add_argument('--output_weight',type=int,default=256,help='Resize后图片宽度')
    parser.add_argument('--data_path',type=str,default='./data/',help='加载数据集的txt路径文件名')
    parser.add_argument('--file_path',type=str,default='./data/',help='数据集根目录')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=200, help='epoch总数')
    parser.add_argument('--batch_size', type=int, default=5, help='mini-batch大小')
    parser.add_argument('--momentum', type=float, default=0.9, help='优化器动量因子')
    parser.add_argument('--weight_decay', type=float, default=0, help='优化器权重衰减值')
    parser.add_argument('--step_size', type=int, default=7, help='学习率更新间隔')
    parser.add_argument('--gamma', type=float, default=0.1, help='学习率衰减因子')
    parser.add_argument('--pretrained', type=int, default=0, help='是否使用预训练模型参数')
    parser.add_argument('--batchnormal', type=int, default=1, help='是否加入batchnormal层')
    parser.add_argument('--viz', type=int, default=1, help='是否使用visdom可视化')
    parser.add_argument('--device_id', type=int, default=0, help='使用GPU的ID')
    # parser.add_argument('--normal',type=str,default='(0.5,0.5,0.5),(0.5,0.5,0.5)',help='归一化均值与标准差')
    parser.add_argument('--decay_scheduler', type=int, default=1, help='是否衰减学习率')
    parser.add_argument('--use_classweight',type=int,default=1,help='是否使用loss类别加权')
    parser.add_argument('--in_channels',type=int,default=3,help='分类网络输入通道数')
    args = parser.parse_args()
    # args.file_path = '/Users/yangweiwei/untitled4/images/yangweiwei/wikiart/**/'

    # args.file_path = os.path.join("/data2/yangweiwei/wikiart", args.dataset_name)
    # args.data_path = './style/'
    logging.info(args)
    print('基本配置信息:\n{}'.format(args))

    # 是否中心切割
    if args.crop:
        transform = transforms.Compose([transforms.Resize(args.input_height),
                                        transforms.CenterCrop((args.output_height, args.output_weight)),
                                        transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize(args.output_height),
                                        transforms.ToTensor()])


    train_set = Dataset(class_num=eval(args.class_num), data_path=os.path.join(args.data_path,'train.txt'), file_path=args.file_path, grayscale=False,
                        transform=transform)
    args.train_n_samples = len(train_set)  # 训练集样本数
    args.train_bincount = train_set.bincount    # 训练集各个类的样本数
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    val_set = Dataset(class_num=eval(args.class_num), data_path=os.path.join(args.data_path,'test.txt'), file_path=args.file_path, grayscale=False,
                      transform=transform)
    args.val_n_samples = len(val_set)   # 验证集样本数
    args.val_bincount = val_set.bincount    # 验证集各个类的样本数
    val_loader = DataLoader(val_set, args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    dataloader = {'train':train_loader,'val':val_loader}

    # 计算分类的类别数量
    if isinstance(eval(args.class_num), int):
        args.class_num = eval(args.class_num)
    elif isinstance(eval(args.class_num), list):
        args.class_num = len(eval(args.class_num))

    # 动态定义网络模型
    model = eval(args.model_name)(args.pretrained, args.output_height, args.class_num, args.in_channels)

    # if args.model_name == 'resnet50':
    #     model = resnet50(args.pretrained, args.output_height,args.class_num)
    # elif args.model_name == 'resnet101':
    #     model = resnet101(args.pretrained, args.output_height,args.class_num)
    # elif args.model_name == 'vgg19':
    #     model = myVgg19(args)

    # if args.pretrained:
    #     #将预训练的参数层修改为不可导，仅训练全连接层,如果不要就注释掉
    #     optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum)
    # else:
    #     optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer_ft = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5,0.999),weight_decay=args.weight_decay)

    # 如果GPU可用
    if args.use_gpu and torch.cuda.is_available():
        model = model.cuda(args.device_id)

    # 学习率衰减
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.step_size, gamma=args.gamma)

    model_ft = train_model(model=model, dataloader=dataloader, args=args, optimizer=optimizer_ft, scheduler=exp_lr_scheduler)
    # save the best model
    model_name = args.model_name+ "_inchannel="+ str(args.in_channels)
    if args.batchnormal:
        model_name += '_bn'
    if args.pretrained :
        model_name += '_pred'
    model_name += '_class=' + str(args.class_num) + '_addclassweight.pth'
    torch.save(model_ft.state_dict(),model_name)