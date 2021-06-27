import cv2 as cv
import glob
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

from model.SegNet import SegNet
from utils.dataset import CityDataset

def predict(net, device, data_dir, save_dir, image_transforms):
    images_path = glob.glob(os.path.join(data_dir, '*.png'))
    net.eval()
    with torch.no_grad():
        for image_path in images_path:
            image = Image.open(image_path)
            image = image.resize((512, 256), Image.BILINEAR)
            image = image_transforms(image)
            image = image.unsqueeze(0)
            image = image.to(device=device, dtype=torch.float32)

            pred = net(image)
            pred = F.softmax(pred, dim=1)
            pred = pred.argmax(1)
            pred = np.array(pred.data.cpu()[0])
            pred = np.where(pred==1, 255, pred)
            pred = np.where(pred==2, 128, pred)
            pred = np.array(pred, dtype=np.uint8)

            image_raw = cv.imread(image_path)
            image_raw = cv.resize(image_raw, (512, 256))
            result = np.concatenate((image_raw, (np.stack((pred,)*3, axis=-1))), axis=1)
            cv.imwrite(os.path.join(save_dir, os.path.basename(image_path)), result)

def eval(net, device, dataset_test):
    test_loader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    net.eval()
    with torch.no_grad():
        test_loss = 0.0
        for image, label in test_loader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            pred = net(image)
            loss = loss_fn(pred, label)
            test_loss += loss.item() * image.size(0)
        test_loss = test_loss / len(test_loader.dataset)
        print('\tTest Loss: {:.6f}'.format(test_loss))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SegNet(3, 3)
    net.to(device=device)
    net.load_state_dict(torch.load('model.pth', map_location=device))

    test_dir = 'D:/pytorch/Segmentation/cityscapes/data/image/test'
    label_ids = [[7], [26, 27, 28, 29]]
    target_size = [512, 256]
    save_dir = 'predict'
    image_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset_test = CityDataset(test_dir, label_ids, target_size, image_transforms)
    eval(net, device, dataset_test)

    test_list = os.listdir(test_dir)
    for dir in test_list:
        data_dir = os.path.join(test_dir, dir)
        predict(net, device, data_dir, save_dir, image_transforms)

if __name__ == '__main__':
    main()