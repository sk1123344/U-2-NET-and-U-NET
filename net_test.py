import torch
import cv2
from unet import UNET
from u2net import U2NET
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd


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


def load_net(name, path):
    #print(torch.load(path).keys())
    if name.lower() == 'unet':
        model = UNET()
        model.load_state_dict(state_dict=torch.load(path))
        return model
    if name.lower() == 'u2net':
        model = U2NET()
        model.load_state_dict(state_dict=torch.load(path))
        return model


def results_output(net_name, net_path, data_path, mask_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_net(net_name, net_path).to(device)
    test_loader = DataLoader(DataInput(data_path, mask_path), shuffle=False, num_workers=0, batch_size=1)
    save_path = 'testing_results_{}'.format(net_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for idx, (image, mask, mask_path) in enumerate(test_loader):
        image, mask = image.to(device), mask.to(device)
        model.eval()
        with torch.no_grad():
            output = model(image)
            save_images(output[0] if net_name == 'u2net' else output, mask_path, save_path)
            iou_calculation_save(output[0] if net_name == 'u2net' else output, mask, mask_path, save_path)


name = 'unet'
model_path = 'pretrained_models/{}_150.pth'.format(name)
test_data_path = 'dataset/test_data'
test_mask_path = 'dataset/test_mask'
results_output(name, model_path, test_data_path, test_mask_path)