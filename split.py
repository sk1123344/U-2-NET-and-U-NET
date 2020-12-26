import os
import cv2

path = ['./train', './test']

#print(True if 'train' in './train' else False)


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


for path_ in path:
    name_list = os.listdir(path_)
    root = os.getcwd()
    if 'train' in path_:
        dir_name = 'train'
        image_dir_name = 'train_data'
        mask_dir_name = 'train_mask'
        make_dir(os.path.join(root, image_dir_name))
        make_dir(os.path.join(root, mask_dir_name))
    else:
        dir_name = 'test'
        image_dir_name = 'test_data'
        mask_dir_name = 'test_mask'
        make_dir(os.path.join(root, image_dir_name))
        make_dir(os.path.join(root, mask_dir_name))
    for name in name_list:
        #print(name.split('.')[-1])
        if name.split('.')[-1] == "png":
            filepath = os.path.join(root, dir_name, name)
            #print(filepath)
            img = cv2.imread(filepath)
            w = img.shape[1]
            img_data = img[:, :int(w / 2), :]
            img_mask = img[:, int(w / 2):, :]
            cv2.imwrite(os.path.join(root, image_dir_name, name), img_data)
            cv2.imwrite(os.path.join(root, mask_dir_name, name), img_mask)