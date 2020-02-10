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
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='3')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06743.jpg') 
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    
    return parser.parse_args()

def load_check(path):
    check = torch.load(path)
    model = getattr(torchvision.models, check['arch'])(pretrained=True)
    # Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = NeuralNetwork(check['input_size'],
                             check['output_size'],
                             check['hidden_layers'],
                             check['drop'])
    model.classifier = classifier

    
    model.classifier.load_state_dict(check['state_dict'])
    
    model.classifier.optimizer = check['optimizer']
    model.classifier.epochs = check['epochs']
    model.classifier.learning_rate = check['learning_rate']
    model.classifier.class_to_idx = check['class_to_idx']

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    img = img.resize((256, 256))
    
    s = img.size
    img = img.crop((s[0]//2 -(224/2), s[1]//2 - (224/2), s[0]//2 +(224/2), s[1]//2 + (224/2)))
    
    img = np.array(img)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = (img - mean)/std
    img = img.transpose(2,0,1)
    
    return img

def predict(image_path, model, topk=5, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    if gpu == 'gpu':
        model = model.cuda()
    else:
        model = model.cpu()
    
    img = process_image(image_path)
    
    img = torch.from_numpy(np.array([img])).float()
    img = Variable(img)
    
    if gpu == 'gpu':
        with torch.no_grad():
            out = model.forward(img.cuda())
    else:
        with torch.no_grad():
            out = model.forward(img)
    
    topProb, topLab = torch.topk(output, topk)
    topProb = topProb.exp()
    topProbArray = topProb.cpu().data.numpy()[0]
    
    index = []
    for i in range(len(model.class_to_idx.items())):
        index.append(list(model.class_to_idx.items())[i][0])
        
    topLabData = topLab.cpu().data.numpy()
    topLabList = topLabData[0].tolist()
    
    topClass = []
    for i in topLabList:
        topClass.append(index[i])
        
    return topProbArray, topClass

def loadCatNames(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names

def main():
    
    args = parse_args()
    
    gpu = args.gpu
    
    model = load_check(args.checkpoint)
    catName = loadCatNames(args.category_names)
    
    img_path = args.filepath
    
    probs, classes = predict(img_path, model, int(args.top_k), gpu)
    
    labs = []
    for i in classes:
        labs.append(catName[str(i)])
                    
    probab = probs
    print('File: ' + img_path)
    
    print(labs)
    print(probab)
    
    i = 0 
    while i < len(labs):
        print("{} with probability {}".format(labs[i], probab[i]))
        i += 1
        

if __name__ == "__main__":
    main()