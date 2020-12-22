import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from unet import UNET
import pandas as pd
import numpy as np


def save_images(image_tensor, mask_path_, save_path):
    image_num = image_tensor.size(0)
    images_ = torch.round(image_tensor.clone().detach()).permute(0, 2, 3, 1).cpu().numpy() * 255
    for i in range(image_num):
        cv2.imwrite(os.path.join(save_path, os.path.basename(mask_path_[i])), cv2.resize(images_[i, :, :, :], dsize=cv2.imread(mask_path_[i]).shape[:2], interpolation=cv2.INTER_NEAREST))


def iou_calculation_save(image_tensor, target, mask_path_, save_path):
    pred = 1 - torch.round(image_tensor.clone().detach()).long()
    target = 1 - torch.round(target.clone().detach()).long()
    intersection = torch.sum(pred & target, dim=(1, 2, 3)).float()
    union = torch.sum(pred | target, dim=(1, 2, 3)).float()
    ious = intersection / union
    txt_path = os.path.join(save_path, 'iou.txt')
    if not os.path.isfile(txt_path):
        df = pd.DataFrame(np.c_[np.array([os.path.basename(x) for x in mask_path_]), ious.detach().cpu().numpy()])
        df.to_csv(txt_path, sep=' ', index=False, header=False)
    else:
        data = pd.read_table(txt_path, sep=' ', header=None).values
        df = pd.DataFrame(np.r_[data, np.c_[np.array([os.path.basename(x) for x in mask_path_]), ious.detach().cpu().numpy()]])
        df.to_csv(txt_path, sep=' ', index=False, header=False)


def bce_loss_all(out, target):
    criterion = nn.BCELoss()
    loss = criterion(out, target)
    return loss


class DataInput(Dataset):
    def __init__(self, data_path, mask_path, resize=512, data_postfix='.png', mask_postfix='.png'):
        self.data_path = self.path_list_obtain(data_path, data_postfix)
        self.mask_path = self.path_list_obtain(mask_path, mask_postfix, self.data_path)
        #print(self.mask_path)
        self.resize = resize

    def __getitem__(self, item):
        image = cv2.resize(cv2.imread(self.data_path[item]) / 255, dsize=(self.resize, self.resize), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(cv2.imread(self.mask_path[item])[:, :, 0] / 255, dsize=(self.resize, self.resize), interpolation=cv2.INTER_NEAREST)
        #print(target.shape)
        return torch.tensor(image, dtype=torch.float).permute(2, 0, 1), torch.tensor(target, dtype=torch.float).unsqueeze(0), self.mask_path[item]

    def __len__(self):
        return len(self.data_path)

    def path_list_obtain(self, path, postfix, path_list=None):
        root = os.getcwd()
        if path_list is None:
            data_path_list = []
            for file in os.listdir(path):
                if os.path.splitext(file)[-1] == postfix:
                    data_path_list.append(os.path.join(root, path, file))
            return data_path_list
        else:
            data_path_list = []
            for path_ in path_list:
                data_path_list.append(os.path.join(root, path, os.path.splitext(os.path.split(path_)[1])[0] + postfix))
            return data_path_list


train_data_path = 'dataset/train_data'
train_mask_path = 'dataset/train_mask'
test_data_path = 'dataset/test_data'
test_mask_path = 'dataset/test_mask'
batch_size = 20
epochs = 1000
save_epoch = 10
num_workers = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data_loader = DataLoader(DataInput(train_data_path, train_mask_path), shuffle=True, num_workers=num_workers, batch_size=batch_size)
test_data_loader = DataLoader(DataInput(test_data_path, test_mask_path), shuffle=False, num_workers=num_workers, batch_size=1)
model = UNET()
m = model
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    m = model.module
model.to(device)

optimizer = torch.optim.Adam(model.parameters())
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.99, patience=10)
scheduler = None
if not os.path.isdir(os.path.join(os.getcwd(), 'models')):
    os.makedirs(os.path.join(os.getcwd(), 'models'))

print('--------start to train--------')
for epoch in range(epochs):
    model.train()
    loss_all = 0
    total = 0
    for idx, (images, targets, path) in enumerate(train_data_loader):
        images = images.to(device)
        targets = targets.to(device)

        out = model(images)
        loss = bce_loss_all(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all += loss.detach().cpu().numpy() * images.size(0)
        total += images.size(0)
    if scheduler is not None:
        scheduler.step(loss_all / total)
    print('[epoch {}|{}]: loss = {}, lr = {}'.format(epoch + 1, epochs, loss_all / total, optimizer.state_dict()['param_groups'][0]['lr']))
    if (epoch + 1) % save_epoch == 0:
        torch.save(m.state_dict(), 'models/unet_{}.pth'.format(epoch + 1))
        folder = os.path.join(os.getcwd(), 'results_unet', 'testing_result_{}'.format(epoch + 1))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        model.eval()
        with torch.no_grad():
            for idx, (images, targets, path) in enumerate(test_data_loader):
                images = images.to(device)
                targets = targets.to(device)

                out = model(images)
                iou_calculation_save(out, targets, path, folder)
                save_images(out, path, folder)
