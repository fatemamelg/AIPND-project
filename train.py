import numpy as np
import matplotlib.pyplot as plt

import os, random

from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models

from PIL import Image
import json
import argparse



resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='vgg13', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.01')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    
    for e in range(epochs):
    train = 0
    val = 1
    
    for phase in [train, val]:
        if phase == train:
            model.train()
        else:
            model.eval()
            
        pass_cnt = 0
    
        for inputs, labels in dataloaders[phase]:
            
            pass_cnt += 1
            
            if gpu == 'gpu':
                model.cuda()
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                model.cpu() 
        
            optimizer.zero_grad()
            
            output = model.forward(inputs)
            loss = criterion(output, labels)
            
            if phase == train:
                loss.backward()
                optimizer.step()
                
            run_loss += loss.item()
            ps = torch.exp(output).data
            equ = (labels.data == ps.max(1)[1])
            acc = equ.type_as(torch.cuda.FloatTensor()).mean()
            
        if phase == train:
            print("\nEpoch: {}/{} ".format(e+1, epochs),
                  "\nTraining Loss: {:.4f}  ".format(run_loss/pass_cnt))
        else:
            print("Validation Loss: {:.4f}  ".format(run_loss/pass_cnt),
              "Accuracy: {:.4f}".format(acc))

        run_loss = 0


def main():
    
    args = parse_args()
    
    rate = float(args.learning_rate)
    hl = int(args.hidden_units)
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_T = transforms.Compose([transforms.RandomRotation(30),
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ])
    
    train_D = datasets.ImageFolder(train_dir, transform = train_T)
    
    train_L = torch.utils.data.DataLoader(train_D, batch_size = 64, shuffle = True)
    
    model = getattr(models, args.arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    if args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu1', nn.ReLU()),
                                  ('hidden_layer1', nn.Linear(hl, 90)),
                                  ('relu2',nn.ReLU()),
                                  ('hidden_layer2',nn.Linear(90,80)),
                                  ('relu3',nn.ReLU()),
                                  ('fc2', nn.Linear(1024, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 500)),
                                  ('drop', nn.Dropout(p=0.6)),
                                  ('relu1', nn.ReLU()),
                                  ('hidden_layer1', nn.Linear(hl, 90)),
                                  ('relu2',nn.ReLU()),
                                  ('hidden_layer2',nn.Linear(90,80)),
                                  ('relu3',nn.ReLU()),
                                  ('fc2', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=rate)
    
    epochs = int(args.epochs)
    class_index = image_datasets[0].class_to_idx
    gpu = args.gpu
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = class_index
    path = args.save_dir 
    
    # TODO: Save the checkpoint 
    model.class_to_idx = image_datasets[0].class_to_idx

    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'arch': 'vgg19',
                  'learning_rate': 0.01,
                  'batch_size': 64,
                  'classifier' : classifier,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')


if __name__ == "__main__":
    main()
    