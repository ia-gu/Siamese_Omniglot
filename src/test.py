# hand-made module
from mydataset import OmniglotTest
from model import Siamese, VGG, ResNet18

# pyTorch
import torch
from torchvision import transforms
from torchvision import transforms
from torch.utils.data import DataLoader

# other
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import random
import logging
import argparse
import yaml

def test(config, model, dir_path):
    
    model.eval()
    labels = None
    device = torch.device('cuda')
    testSet = OmniglotTest('../../data/omniglot/images_evaluation', transform=transforms.ToTensor(), times=config['shots'], way=config['way'])
    testLoader = DataLoader(testSet, **config['test_loader'])

    right, error = 0, 0
    for _, (test1, test2, label) in enumerate(testLoader, 1):
        test1 = test1.to(device, non_blocking=True)
        test2 = test2.to(device, non_blocking=True)
        output = model.forward(test1, test2).data.cpu().numpy()
        pred = np.argmax(output)
        if pred == 0:
            right += 1
        else: error += 1

        if(labels == None):
            labels = label
            outputs = output
        else:
            labels = torch.cat((labels, label), 0)
            outputs = np.concatenate([outputs, output])

    print('*'*70)
    print(f'Test result   correct:{right}   error:{error}   accuracy:{right/(right+error):.3f}')
    print('*'*70)
    logging.error(f'Test result   correct:{right}   error:{error}   accuracy:{right/(right+error):.3f}')

    # ROC curve
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, outputs, drop_intermediate=False)
    fig = plt.figure()
    plt.plot(fpr, tpr, marker='o', markersize=1)
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    fig.savefig(dir_path+'roc_curve.png')

# Only Test
if __name__ == '__main__':
    # seed
    seed = 9999
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # get config
    def get_args():
        parser = argparse.ArgumentParser(description='YAML')
        parser.add_argument('config_path', type=str, help='.yaml')
        parser.add_argument('dir_path', type=str)
        parser.add_argument('weight_path', type=str)
        args = parser.parse_args()
        return args
    args = get_args()
    dir_path = args.dir_path
    weight_path = args.weight_path

    logging.basicConfig(level=logging.ERROR, filename=dir_path+'result.txt', format="%(message)s")

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    if(config['model'] == 'Siamese'):
        model=Siamese()
    elif(config['model'] == 'VGG'):
        model = VGG(64)
    elif(config['model'] == 'ResNet'):
        model = ResNet18()
    
    device = torch.device('cuda')
    model = model.to(device)
    params = torch.load(weight_path, map_location=device)
    model.load_state_dict(params)

    test(config, model, dir_path)