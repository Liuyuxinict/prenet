from __future__ import print_function
import os
from PIL import Image
import torch.utils.data as data
import os
import PIL
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import re
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
EPOCH         = 200            # number of times for each run-through
BATCH_SIZE    = 4            # number of images for each epoch
LEARNING_RATE = 0.0001             # default learning rate
GPU_IN_USE    = torch.cuda.is_available()  # whether using GPU
DIR_TRAIN_IMAGES   = '/workdir/wangzhiling02/train_finetune.txt'
#DIR_TRAIN_IMAGES   = "/home/vipl/lyx/train_full.txt"
DIR_TEST_IMAGES    = '/workdir/wangzhiling02/test_finetune.txt'
#DIR_TEST_IMAGES    = "/home/vipl/lyx/test_full.txt"
Image_path = "/workdir/wangzhiling02/Food2k_complete/"
#Image_path = "/home/vipl/lizhuo/dataset_food/food101/images/"
#NUM_CATEGORIES     = 500
NUM_CATEGORIES     = 2000
#WEIGHT_PATH= '/home/vipl/lyx/resnet50.pth'
WEIGHT_PATH = '/home/hadoop-mtcv/cephfs/data/wangzhiling02/baseline_code_1/resnet/model/food2k_resnet50_0.0001.pth'

checkpoint = ''
useJP = False  #use Jigsaw Patches during PMG food2k_448_from2k_only_cengnei
usecheckpoint = True
checkpath = "./food2k_448_from2k_only_cengnei/model.pth"

useAttn = True

normalize = transforms.Normalize(mean=[0.5457954,0.44430383,0.34424934],
                                  std=[0.23273608,0.24383051,0.24237761])
train_transforms = transforms.Compose([
                     transforms.RandomHorizontalFlip(p=0.5), # default value is 0.5
                     transforms.RandomRotation(degrees=15),
                     transforms.ColorJitter(brightness=0.126,saturation=0.5),
                     transforms.Resize((550, 550)),
                     transforms.RandomCrop(448),
                     transforms.ToTensor(),
                     normalize
                  ])

# transforms of test dataset
test_transforms = transforms.Compose([
                    transforms.Resize((550, 550)),
                    transforms.CenterCrop((448,448)),
                    transforms.ToTensor(),
                    normalize
                  ])

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, txt_dir, transform=None, target_transform=None, loader=My_loader):
        data_txt = open(txt_dir, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split(' ')
            imgs.append((words[0], int(words[1].strip())))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = My_loader

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        # label = list(map(int, label))
        # print label
        # print type(label)
        #img = self.loader('/home/vipl/llh/food101_finetuning/food101_vgg/origal_data/images/'+img_name.replace("\\","/"))
        img = self.loader(Image_path + img_name)

        # print img
        if self.transform is not None:
            img = self.transform(img)
            # print img.size()
            # label =torch.Tensor(label)

            # print label.size()
        return img, label
        # if the label is the single-label it can be the int
        # if the multilabel can be the list to torch.tensor

train_dataset = MyDataset(txt_dir=DIR_TRAIN_IMAGES , transform=train_transforms)
test_dataset = MyDataset(txt_dir=DIR_TEST_IMAGES , transform=test_transforms)
train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=BATCH_SIZE//2,  shuffle=False, num_workers=2)
print('Data Preparation : Finished')

def train(nb_epoch, trainloader, testloader, batch_size, store_name, start_epoch, net,optimizer,exp_lr_scheduler):
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)


    CELoss = nn.CrossEntropyLoss()
    KLLoss = nn.KLDivLoss(reduction="batchmean")


    max_val_acc = 0
    #val_acc, val5_acc, _, _, val_loss = test(net, CELoss, batch_size, testloader)

    for epoch in range(start_epoch, nb_epoch):

        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0
        batch_idx = 0
        u1 = 1
        u2 = 0.5
        for (inputs, targets) in tqdm(trainloader):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            # Step 1
            optimizer.zero_grad()
            #inputs1 = jigsaw_generator(inputs, 8)
            inputs1 = None
            if useJP:
                _, _, _, _, output_1, _, _ = net(inputs1,False)
            else:
                _, _, _, _, output_1, _, _ = net(inputs, False)
                #print(output_1.shape)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()

            # Step 2
            optimizer.zero_grad()
            #inputs2 = jigsaw_generator(inputs, 4)
            inputs2 = None
            if useJP:
                _, _, _, _, _, output_2, _, = net(inputs2,False)
            else:
                _, _, _, _, _, output_2, _, = net(inputs, False)
                #print(output_2.shape)
            loss2 = CELoss(output_2, targets) * 1
            loss2.backward()
            optimizer.step()

            # Step 3
            optimizer.zero_grad()
            #inputs3 = jigsaw_generator(inputs, 2)
            inputs3 = None
            if useJP:
                _, _, _, _, _, _, output_3 = net(inputs3,False)
            else:
                _, _, _, _, _, _, output_3 = net(inputs, False)
                #print(output_3.shape)
            loss3 = CELoss(output_3, targets) * 1
            loss3.backward()
            optimizer.step()


            optimizer.zero_grad()
            x1, x2, x3, output_concat, _, _, _ = net(inputs,useAttn)
            concat_loss = CELoss(output_concat, targets) * 2


            #loss4 = -KLLoss(F.softmax(x1, dim=1), F.softmax(x2, dim=1)) / batch_size
            #loss5 = -KLLoss(F.softmax(x1, dim=1), F.softmax(x3, dim=1)) / batch_size
            loss6 = -KLLoss(F.softmax(x2, dim=1), F.softmax(x1, dim=1))
            #loss7 = -KLLoss(F.softmax(x2, dim=1), F.softmax(x3, dim=1)) / batch_size
            loss8 = -KLLoss(F.softmax(x3, dim=1), F.softmax(x1, dim=1))
            loss9 = -KLLoss(F.softmax(x3, dim=1), F.softmax(x2, dim=1))

            Klloss = loss6 + loss8 + loss9

            totalloss = u1 * concat_loss + u2 * Klloss
            totalloss.backward()
            optimizer.step()

            #  training log
            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()

            if batch_idx % 10 == 0:
                print(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                    train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))
            batch_idx += 1

        exp_lr_scheduler.step()

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                train_loss4 / (idx + 1)))

        val_acc, val5_acc, val_acc_com, val5_acc_com, val_loss = test(net, CELoss, batch_size, testloader,useAttn)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(net, './' + store_name + '/model.pth')
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write(
                'Iteration %d, top1 = %.5f, top5 = %.5f, top1_combined = %.5f, top5_combined = %.5f, test_loss = %.6f\n' % (
                    epoch, val_acc, val5_acc, val_acc_com, val5_acc_com, val_loss))

net = load_model('resnet50_pmg',pretrain=False,require_grad=True,num_class=NUM_CATEGORIES)
net.fc = nn.Linear(2048, 2000)
state_dict = {}
pretrained = torch.load(WEIGHT_PATH)

for k, v in net.state_dict().items():
    if k[9:] in pretrained.keys() and "fc" not in k:
        state_dict[k] = pretrained[k[9:]]
    elif "xx" in k and re.sub(r'xx[0-9]\.?',".", k[9:]) in pretrained.keys():
        state_dict[k] = pretrained[re.sub(r'xx[0-9]\.?',".", k[9:])]
    else:
        state_dict[k] = v
        print(k)

net.load_state_dict(state_dict)
net.fc = nn.Linear(2048,NUM_CATEGORIES)

ignored_params = list(map(id, net.features.parameters()))
new_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
optimizer = optim.SGD([
    {'params': net.features.parameters(), 'lr': LEARNING_RATE*0.1},
    {'params': new_params, 'lr': LEARNING_RATE}
],
    momentum=0.9, weight_decay=5e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
for p in optimizer.param_groups:
    outputs = ''
    for k, v in p.items():
        if k is 'params':
            outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
        else:
            outputs += (k + ': ' + str(v).ljust(10) + ' ')
    print(outputs)

cudnn.benchmark = True
net.cuda()
net = nn.DataParallel(net)

if usecheckpoint:
    #net.load_state_dict(torch.load(checkpath))
    net.module.load_state_dict(torch.load(checkpath).module.state_dict())
    print('load the checkpoint')


train(nb_epoch=200,             # number of epoch
         trainloader=train_loader,
         testloader=test_loader,
         batch_size=BATCH_SIZE,         # batch size
         store_name='food2k_448_from2k_only_cengnei',     # folder for output
         start_epoch=0,
         net=net,
        optimizer = optimizer,
        exp_lr_scheduler=exp_lr_scheduler)         # the start epoch number when you resume the training

