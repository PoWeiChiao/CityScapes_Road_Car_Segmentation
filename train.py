import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model.SegNet import SegNet
from utils.dataset import CityDataset
from utils.logger import Logger

def train(net, device, dataset_train, dataset_val, batch_size=4, epochs=10, lr=0.00001):
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True)

    optimizer = optim.RMSprop(params=net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = np.inf
    log_train = Logger('log_train.txt')
    log_val   = Logger('log_val.txt')
    writer = SummaryWriter('runs/exp_1')

    for epoch in range(epochs):
        loss_train = 0.0
        loss_val   = 0.0
        print('running epoch {}'.format(epoch))

        net.train()
        for image, label in tqdm(train_loader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            pred = net(image)
            loss = loss_fn(pred, label)
            loss_train += loss.item() * image.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        for image, label in tqdm(val_loader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            pred = net(image)
            loss = loss_fn(pred, label)
            loss_val += loss.item() * image.size(0)

        loss_train = loss_train / len(train_loader.dataset)
        loss_val   = loss_val / len(val_loader.dataset)

        print('\tTraining loss: {:.6f} \tValidation loss: {:.6f}'.format(loss_train, loss_val))
        writer.add_scalar('train loss', loss_train, epoch)
        writer.add_scalar('val loss', loss_val, epoch)
        log_train.write_epoch_loss(epoch, loss_train)
        log_val.write_epoch_loss(epoch, loss_val)

        if loss_val <= best_loss:
            torch.save(net.state_dict(), 'model.pth')
            best_loss = loss_val
            print('model saved')
        if epoch >= 40:
            torch.save(net.state_dict(), 'model_' + str(epoch) + '.pth')
    writer.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device {}'.format(device))

    train_dir = 'D:/pytorch/Segmentation/cityscapes/data/image/train'
    val_dir   = 'D:/pytorch/Segmentation/cityscapes/data/image/val'
    label_ids = [[7], [26, 27, 28, 29]]
    target_size = [512, 256]
    image_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset_train = CityDataset(train_dir, label_ids, target_size, image_transforms)
    dataset_val   = CityDataset(val_dir, label_ids, target_size, image_transforms)
    
    net = SegNet(3, 3)
    net.to(device=device)

    train(net, device, dataset_train, dataset_val, batch_size=4, epochs=50, lr=0.00001)

if __name__ == '__main__':
    main()