import os,torch
import torchvision.transforms as transforms
from resnet import *
from vggnet import *
from utils import *
from dataset import Dataset
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
val_set = Dataset(class_num=2, data_path=os.path.join('./data', 't.txt'),
                  file_path='./data', grayscale=False, transform=transform)
val_loader = DataLoader(val_set, 1, shuffle=False, num_workers=4, drop_last=True)

# in_channels = 3
# model = vgg16_bn(0, 256, 2, in_channels)
# model.load_state_dict(torch.load('./experiment/vgg16_bn_inchannel=3_class_weight_SGD/vgg16_bn_inchannel=3_bn_class=2_addclassweight.pth'))

in_channels = 5
model = resnet34(0, 256, 2, in_channels)
model.load_state_dict(torch.load('./experiment/resnet34_inchannel=5_class_weight_SGD/resnet34_inchannel=5_bn_class=2_addclassweight.pth'))

model.cuda()
input_img = torch.Tensor(1, in_channels, 256, 256)
input_label = torch.LongTensor(1)
val_running_corrects = 0
val_running_total = 0

for i, (imgs, marks, labels) in enumerate(val_loader):
    # 合并输入通道
    if in_channels > 3:
        inputs = input_img.copy_(torch.cat((imgs, *marks), 1))
    else:
        inputs = input_img.copy_(imgs)

    inputs = inputs.cuda()
    labels = labels.cuda()
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    # print(labels,preds)
    val_running_corrects += torch.Tensor(
        [float(torch.sum((preds == j) & (labels == j))) for j in range(2)])
    val_running_total += torch.Tensor([float(torch.sum(labels == j)) for j in range(2)])

TP = val_running_corrects[1]
TN = val_running_corrects[0]
FP = val_running_total[0] - val_running_corrects[0]
FN = val_running_total[1] - val_running_corrects[1]

print('recall: ' + str(recall(TP,TN,FP,FN)))
print('precision: ' + str(precision(TP, TN, FP, FN)))
print('acc: ' + str(accuracy(TP,TN,FP,FN)))
print('F1: '+ str(F1_measure(TP,TN,FP,FN)))

