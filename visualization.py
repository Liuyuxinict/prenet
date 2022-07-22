import torch
import torch.nn as nn
import torch.optim as optim

import cv2
import numpy as np
import torchvision
from torchvision import datasets

# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import load_model

check = False
load_pretrained = False

#model = ResNet_aac(fb=64, n_label = 11, model_size=152,kernel_size=3,stride=2,dk=1,dv=1, Nh=8,shape=224,relative = False)
model = load_model('resnet50_pmg',pretrain=False,require_grad=True,num_class=101)
model.cuda()
model = nn.DataParallel(model)
#model = resnet152()
pointlist=[]

if load_pretrained:
    model_dict = model.state_dict()
    #pretrained_dict = torch.load(args.pretrainedmodel)
    pretrained_dict = torch.load("D:/resnet50.pth")
    # pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    state_dict = {}

    for k, v in model_dict.items():
        # if k in dict1.keys():
        if k in pretrained_dict.keys() and "fc" not in k:
            #state_dict[k] = pretrained_dict[dict1[k]]
            state_dict[k] = pretrained_dict[k]

        else:
            state_dict[k] = v
            print(k)

if check:
    #filename = "best_model_"
    #checkpoint = torch.load('./checkpoint/' + filename + 'ckpt.t7')
    #checkpoint = torch.load('unprebest.t7')
    model.module.load_state_dict(torch.load("./food101/model.pth").module.state_dict())


model.eval()
test_img = "G:/images/apple_pie/116697.jpg"
img = Image.open(test_img).convert('RGB')
transform_test = transforms.Compose([
            transforms.Resize(size=(299, 299)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5457954,0.44430383,0.34424934), (0.23273608,0.24383051,0.24237761))
        ])
img1 = transform_test(img).reshape([1,3,224,224])
model(img1,True)
#print(model(img1,True).shape)


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print (xy)
        pointlist.append((x,y))
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,0), thickness = 1)
        cv2.imshow("Image", img)
        return


test_img = "G:/images/apple_pie/116697.jpg"
#显示原图
img = cv2.imread(test_img)

img = cv2.resize(img,(224,224))
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", on_EVENT_LBUTTONDOWN)
img_raw=img.copy()
cv2.imshow("Image", img)

cv2.waitKey(0)




#显示normalize之后的图
img = Image.open(test_img).convert('RGB')
transform_test = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5457954,0.44430383,0.34424934), (0.23273608,0.24383051,0.24237761))
        ])
img1 = transform_test(img)
img = img1.transpose(0,1).transpose(1,2)
img = img.numpy()
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow("Image", img)
cv2.waitKey (0)



#显示热力图
AAC_mat = np.load(r"D:\PMG\visual\matrix14.npy")
depth = AAC_mat.shape[-1]
height = int(np.sqrt(depth))
AAC_mat = np.reshape(AAC_mat,[8,depth,depth])
#print(AAC_mat.shape)



isfirst1 = True
imgs = None
imgs1 = None
for item in pointlist:
    isfirst = True
    x,y = item
    x/=224/height
    y/=224/height
    mat = AAC_mat[:,int(x*height+y),:]
    #print(mat.shape)
    #for i in range(0,8):
    result = mat

    result = result.reshape([8,height,height])
    for i in range(0,8):
        result = mat[i:]
        img = result
        #img = cv2.resize(result, (224, 224))
        #img = img*255

        heatmap = img / np.max(img)
        heatmap = np.uint8(255 * heatmap)
        w=heatmap
        w = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 转化为jet 的colormap
        #w = heatmap
        w = cv2.resize(w,(224,224))
        #x = img_raw*0.5+w*0.5   # 权重自己定
        x =w
        x = x.astype(np.uint8)
        #print(x.shape)
        if isfirst:
            imgs = x
            isfirst = False
        else:
            print(imgs.shape)
            print(x.shape)
            imgs = np.hstack([imgs, x])
        #print(imgs.shape)

    if isfirst1:
        imgs1 = imgs
        isfirst1 = False
    else:
        imgs1 = np.vstack([imgs1, imgs])

print(imgs1.shape)
cv2.imshow("mutil_pic", imgs1)
cv2.waitKey(0)
