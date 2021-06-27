import glob
import numpy as np
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CityDataset(Dataset):
    def __init__(self, image_dir, label_ids, target_size, image_transforms):
        self.image_dir = image_dir
        self.label_ids = label_ids
        self.target_size = target_size
        self.image_transforms = image_transforms

        list_dir = os.listdir(image_dir)
        self.image_list = []
        self.label_list = []
        for dir in list_dir:
            images = glob.glob(os.path.join(image_dir, dir, '*png'))
            for image in images:
                self.image_list.append(image)
                self.label_list.append((image[:-15] + 'gtFine_labelIds.png').replace('image', 'label'))
        self.image_list.sort()
        self.label_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        label = Image.open(self.label_list[index])

        image = image.resize((self.target_size[0], self.target_size[1]), Image.BILINEAR)
        label = label.resize((self.target_size[0], self.target_size[1]), Image.NEAREST)

        isFlip = random.random()
        if isFlip > 0.5:
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
            label = label.transpose(method=Image.FLIP_LEFT_RIGHT)

        image = self.image_transforms(image)

        label = np.array(label)
        shape = label.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                exist = False
                for idx, ids in enumerate(self.label_ids):
                    if label[i][j] in ids:
                        exist = True
                        label[i][j] = idx + 1
                        break
                if not exist:
                    label[i][j] = 0
        label = torch.from_numpy(label)

        return image, label

def main():
    image_dir = os.path.join('D:/pytorch/Segmentation/cityscapes/data/image/train')
    label_ids = [[7], [26, 27, 28, 29]]
    image_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CityDataset(image_dir, label_ids, image_transforms)
    print(dataset.__len__())

if __name__ == '__main__':
    main()