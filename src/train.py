# hand-made module
from mydataset import OmniglotTrain
from model import Siamese, VGG, ResNet18, PretrainedResNet
from earlystopping import EarlyStopping
from test import test
# PyTorch
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchinfo import summary
# other
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import random
import logging
import argparse
import yaml
import csv

if __name__ == '__main__':
    # get config
    def get_args():
        parser = argparse.ArgumentParser(description='YAML')
        parser.add_argument('config_path', type=str, help='.yaml')
        args = parser.parse_args()
        return args
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # use GPU
    device = torch.device('cuda')
    
    # log
    file_name = os.path.basename(__file__)
    config_name = config['config_name']
    file_name = file_name.replace('.py', '')
    dir_path = 'result/' + config['model'] +'/' + file_name + '/' + config_name + '/'
    if(not os.path.exists(dir_path)):
        os.makedirs(dir_path)

    # augument
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    logging.basicConfig(level=logging.ERROR, filename=dir_path+'result.txt', format="%(message)s")
    logging.error('\n'+file_name+'.py')

    # seed
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # dataset
    trainSet = OmniglotTrain('../../data/omniglot/images_background', 'train', transform=data_transforms)
    validSet = OmniglotTrain('../../data/omniglot/images_background', 'validation', transform=data_transforms)
    trainLoader = DataLoader(trainSet, **config['train_loader'])
    validLoader = DataLoader(validSet, **config['valid_loader'])

    # loss
    loss_fn = nn.BCEWithLogitsLoss(size_average=True)

    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
    
    # define model
    if(config['model'] == 'Siamese'):
        model = Siamese()
    elif(config['model'] == 'VGG'):
        model = VGG(64)
    elif(config['model'] == 'ResNet'):
        model = ResNet18()
    logging.error(config['model'])

    model.to(device)
    model.apply(initialize_weights)
    model = nn.DataParallel(model)

    # optimizer
    def make_optimizer(params, name, **kwargs):
        return optim.__dict__[name](params, **kwargs)
    optimizer = make_optimizer(model.parameters(),**config['optimizer'])

    # result history
    history = {
        'train_loss': ['train_loss'],
        'train_acc' : ['train_acc'],
        'valid_loss': ['valid_loss'],
        'valid_acc' : ['valid_acc']
    }
    train_error = train_correct = valid_error = valid_correct = 0
    train_loss = 0
    train_acc = 0
    time_start = time.time()
    earlystopping = EarlyStopping(config['patience'], path = dir_path+'weight_best.pth')

    print('start train phase')

    # start training
    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
        # set model train_mode
        model.train()
        if batch_id > config['iteration']:
            break
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model.forward(img1, img2)
        loss = loss_fn(output,label)
        output = output.data.cpu().numpy()

        # calculate accuracy
        for i in range(len(output)):
            if((output[i]>0.5 and label[i]==1) or (output[i]<=0.5 and label[i]==0)):
                train_correct+=1
            else:
                train_error+=1

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # validation phase
        if batch_id % config['valid_step'] == 0:
            print(f'[{batch_id}]   train_acc:{train_correct*100/(train_correct+train_error):.3f}%   train_loss:{train_loss/config["valid_step"]:.3f}   time_lapsed:{time.time()-time_start:.3f}')
            time_start = time.time()
            valid_correct = valid_error = valid_loss = 0

            for v_batch_id, (v_img1, v_img2, v_label) in enumerate(validLoader, 1):
                # set model valid mode
                model.eval()
                with torch.no_grad():
                    v_img1 = v_img1.to(device, non_blocking=True)
                    v_img2 = v_img2.to(device, non_blocking=True)
                    v_label = v_label.to(device, non_blocking=True)
                    
                    v_out = model.forward(v_img1, v_img2)
                    loss = loss_fn(v_out, v_label)
                    v_out = v_out.data.cpu().numpy()
                    for i in range(len(v_out)):
                        if((v_out[i]>0.5 and v_label[i]==1) or (v_out[i]<=0.5 and v_label[i]==0)):
                            valid_correct += 1
                        else:
                            valid_error += 1
                    
                    valid_loss += loss.item()
                if(v_batch_id>=config['valid_step']):
                    break
            print(f'[{batch_id}]   valid_acc:{valid_correct*100/(valid_correct+valid_error):.3f}%   valid_loss:{valid_loss/config["valid_step"]:.3f}   time_lapsed:{time.time()-time_start:.3f}')
            logging.error(f'[{batch_id}]   train_acc:{train_correct/(train_correct+train_error):.3f}   train_loss:{train_loss/config["valid_step"]:.3f}   valid_acc:{valid_correct/(valid_correct+valid_error):.3f}   valid_loss:{valid_loss/config["valid_step"]:.3f}')
            
            history['train_acc'].append(train_correct/(train_correct+train_error))
            history['train_loss'].append(round(train_loss/config['valid_step'],3))
            history['valid_acc'].append(valid_correct/(valid_correct+valid_error))
            history['valid_loss'].append(round(valid_loss/config['valid_step'],3))

            train_correct=train_error=0
            train_loss = 0
            time.start = time.time()
            
            # EarlyStopping
            earlystopping((valid_loss), model) 
            if earlystopping.early_stop: 
                print("Early Stopping!")
                break
    
    # save weight
    logging.error(optimizer)
    torch.save(model.module.state_dict(), dir_path+'weight.pth')

    # save csv
    with open(dir_path+'result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(history['train_acc'])
        writer.writerow(history['train_loss'])
        writer.writerow(history['valid_acc'])
        writer.writerow(history['valid_loss'])
    history['train_acc'].remove('train_acc')
    history['train_loss'].remove('train_loss')
    history['valid_acc'].remove('valid_acc')
    history['valid_loss'].remove('valid_loss')

    # Accuracy
    fig = plt.figure()
    plt.plot(history['train_acc'])
    plt.plot(history['valid_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.legend(['train acc', 'validation acc'], loc='lower right')
    plt.grid(True)
    fig.savefig(dir_path+'acc.png')

    # Loss
    fig = plt.figure()
    plt.plot(history['train_loss'])
    plt.plot(history['valid_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Itaration')
    plt.legend(['train loss', 'validation loss'], loc='upper right')
    plt.grid(True)
    fig.savefig(dir_path+"loss.png")

    # test model
    test(config, model, dir_path)