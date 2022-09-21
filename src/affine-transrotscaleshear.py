# hand-made module
from mydataset import OmniglotTrain
from model import Siamese, VGG, ResNet18
from earlystopping import EarlyStopping
from test import test
# PyTorch
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torchvision import transforms
from torch.utils.data import DataLoader
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
        parser = argparse.ArgumentParser(description='YAMLありの例')
        parser.add_argument('config_path', type=str, help='設定ファイル(.yaml)')
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

    # augumentation
    data_transforms = transforms.Compose([
        transforms.RandomAffine((-1*config['deg_min'],config['deg_max']), (0.3, 0.3), (0.8, 1.2), (-2, 2)),
        transforms.ToTensor()
    ])
    logging.basicConfig(level=logging.ERROR, filename=dir_path+'result.txt', format="%(message)s")
    logging.error('\n'+file_name+'.py')

    seed = config['seed']

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    # dataset
    trainSet = OmniglotTrain('../../data/omniglot/images_background', 'train', transform=data_transforms)
    validSet = OmniglotTrain('../../data/omniglot/images_background', 'validation', transform=data_transforms)
    trainLoader = DataLoader(trainSet, **config['train_loader'], worker_init_fn=worker_init_fn, generator=g)
    validLoader = DataLoader(validSet, **config['valid_loader'], worker_init_fn=worker_init_fn, generator=g)

    loss_fn = nn.BCEWithLogitsLoss(size_average=True)

    def initialize_weights(m):
        if isinstance(m, nn.Conv2d): # Convolution層が引数に渡された場合
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu') # kaimingの初期化
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)   # bias項は0に初期化
        elif isinstance(m, nn.BatchNorm2d):         # BatchNormalization層が引数に渡された場合
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):              # 全結合層が引数に渡された場合
            nn.init.kaiming_normal_(m.weight.data)  # kaimingの初期化
            nn.init.constant_(m.bias.data, 0)       # biasは0に初期化
    
    # define model
    if(config['model'] == 'Siamese'):
        model=Siamese()
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
    optimizer.zero_grad()

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
    earlystopping = EarlyStopping(config['patience'], verbose=False, path = dir_path+'weight_best.pth')

    print('start')

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
        for i in range(len(output)):
            if((output[i]>0.5 and label[i]==1) or (output[i]<=0.5 and label[i]==0)):
                train_correct+=1
            else:
                train_error+=1

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_id % config['valid_step'] == 0:
            print(f'[{batch_id}]   train_acc:{train_correct*100/(train_correct+train_error):.3f}%   train_loss:{train_loss/config["valid_step"]:.3f}   time_lapsed:{time.time()-time_start:.3f}')
            time_start = time.time()

            valid_correct = valid_error = valid_loss = 0

            # validation & show process
            for v_batch_id, (v_img1, v_img2, v_label) in enumerate(validLoader, 1):
                # set model vevaluation mode
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
                    
                    valid_loss+=loss.item()
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
    
    # save final weight
    logging.error(optimizer)
    torch.save(model.module.state_dict(), dir_path+'weight.pth')

    # save as csv
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