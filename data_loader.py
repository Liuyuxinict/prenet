import torch
import PIL
from PIL import Image
import torch.utils.data as data
from torchvision import datasets, transforms

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, txt_dir, image_path, transform=None, target_transform=None, loader=My_loader):
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
        self.image_path = image_path

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        # label = list(map(int, label))
        # print label
        # print type(label)
        #img = self.loader('/home/vipl/llh/food101_finetuning/food101_vgg/origal_data/images/'+img_name.replace("\\","/"))
        img = self.loader(self.image_path + img_name)

        # print img
        if self.transform is not None:
            img = self.transform(img)
            # print img.size()
            # label =torch.Tensor(label)

            # print label.size()
        return img, label
        # if the label is the single-label it can be the int
        # if the multilabel can be the list to torch.tensor

def load_data(image_path, train_dir, test_dir, batch_size):
    normalize = transforms.Normalize(mean=[0.5457954, 0.44430383, 0.34424934],
                                     std=[0.23273608, 0.24383051, 0.24237761])
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # default value is 0.5
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.126, saturation=0.5),
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448),
        transforms.ToTensor(),
        normalize
    ])

    # transforms of test dataset
    test_transforms = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        normalize
    ])
    train_dataset = MyDataset(txt_dir=train_dir, image_path=image_path, transform=train_transforms)
    test_dataset = MyDataset(txt_dir=test_dir, image_path=image_path, transform=test_transforms)
    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size//2,  shuffle=False, num_workers=0)
    return train_dataset, train_loader, test_dataset, test_loader
