import torch.utils.data as data
import os
import PIL.Image as Image
import cv2

class SaltDataset(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        # img_num = len(os.listdir(os.path.join(root, 'images')))
        img_list = os.listdir(os.path.join(root, 'images'))
        mask_list = os.listdir(os.path.join(root, 'masks'))
        imgs = []
        for file, mask in zip(img_list, mask_list):
            imgs.append([file, mask])

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = cv2.imread(os.path.join('data/images', y_path))

        img_y = cv2.imread(os.path.join('data/masks', y_path))
        if self.transform is not None:
            img_x = self.transform(img_x)

        if self.target_transform(img_y) is not None:
            img_y = self.target_transform(img_y)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


class TestDataset(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        # img_num = len(os.listdir(os.path.join(root, 'images')))
        img_list = os.listdir(os.path.join(root, 'images'))
        imgs = []
        for i, pic in enumerate(img_list):
            imgs.append(pic)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = cv2.imread(os.path.join('test/images', x_path))


        if self.transform is not None:
            img_x = self.transform(img_x)


        return img_x

    def __len__(self):
        return len(self.imgs)

