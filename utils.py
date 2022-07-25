import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from tqdm import tqdm
from model import *
from Resnet import *

def load_model(model_name, pretrain=True, require_grad=True, num_class=1000, pretrained_model=None):
    print('==> Building model..')
    if model_name == 'resnet50':
        net = resnet50(pretrained=pretrain, path=pretrained_model)
        #for param in net.parameters():
            #param.requires_grad = require_grad
        net = PRENet(net, 512, num_class)

    return net

def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


def test(net, criterion, batch_size, testloader,useattn):
    net.eval()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects5 = 0

    val_en_corrects1 = 0
    val_en_corrects2 = 0
    val_en_corrects5 = 0
    batch_idx = 0
    for (inputs, targets) in tqdm(testloader):
        idx = batch_idx
        with torch.no_grad():
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            _, _, _, output_concat, output1, output2, output3 = net(inputs,useattn)
            #print(np.argmax(F.softmax(output_concat, dim=1).cpu().numpy(),axis=1))
            #input()
            #continue
            outputs_com = output1 + output2 + output3 + output_concat

            #print(np.argmax(F.softmax(output1, dim=1).cpu().numpy(),axis=1))
            #input()
            loss = criterion(output_concat, targets)
            test_loss += loss.item()
            _, top3_pos = torch.topk(output_concat.data, 5)
            _, top3_pos_en = torch.topk(outputs_com.data, 5)

            total += targets.size(0)

            batch_corrects1 = torch.sum((top3_pos[:, 0] == targets)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == targets)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == targets)).data.item()
            batch_corrects4 = torch.sum((top3_pos[:, 3] == targets)).data.item()
            batch_corrects5 = torch.sum((top3_pos[:, 4] == targets)).data.item()
            val_corrects5 += (batch_corrects5 + batch_corrects4 + batch_corrects3 + batch_corrects2 + batch_corrects1)

            batch_corrects1 = torch.sum((top3_pos_en[:, 0] == targets)).data.item()
            val_en_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos_en[:, 1] == targets)).data.item()
            val_en_corrects2+= (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos_en[:, 2] == targets)).data.item()
            batch_corrects4 = torch.sum((top3_pos_en[:, 3] == targets)).data.item()
            batch_corrects5 = torch.sum((top3_pos_en[:, 4] == targets)).data.item()
            val_en_corrects5 += (batch_corrects5 + batch_corrects4 + batch_corrects3 + batch_corrects2 + batch_corrects1)

            batch_idx += 1
    test_acc = val_corrects1 / total
    test5_acc = val_corrects5 / total
    test_acc_en = val_en_corrects1 / total
    test5_acc_en = val_en_corrects5 / total
    test_loss = test_loss / (idx + 1)
    return test_acc, test5_acc, test_acc_en, test5_acc_en, test_loss
    #return test_acc, test5_acc, test_loss


