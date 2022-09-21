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

# データローダのseed固定
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# テスト実行
def test(config, model, dir_path):
    # 宣言しなおさないとダメ
    model.eval()
    labels = None
    device = torch.device('cuda')
    g = torch.Generator()
    g.manual_seed(config['seed'])
    testSet = OmniglotTest('../../data/omniglot/images_evaluation', transform=transforms.ToTensor(), times=config['shots'], way=config['way'])
    testLoader = DataLoader(testSet, **config['test_loader'], worker_init_fn=worker_init_fn(config['seed']),generator=g)

    right, error = 0, 0
    for _, (test1, test2, label) in enumerate(testLoader, 1):
        test1 = test1.to(device, non_blocking=True)
        test2 = test2.to(device, non_blocking=True)
        # 85.8
        # for i in range(20):
        #     test1[i] = test1[0]

        # HACK  pytorchのモデルにしたら下がりすぎ
        # print or summaryで確認が必要？
        # self_model: 0.953
        # PyTorch_model: 0.917
        # Pretrained_model: 0.915
        # initialized_model: 0.912
        output = model.forward(test1, test2).data.cpu().numpy()
        pred = np.argmax(output)
        if pred == 0:
            right += 1
        else: error += 1

        # ROC曲線に必要
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

    # ROC曲線
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, outputs, drop_intermediate=False)
    fig = plt.figure()
    plt.plot(fpr, tpr, marker='o', markersize=1)
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    fig.savefig(dir_path+'roc_curve.png')

# config, ログpath，重みpathを渡せば，重みを読み込みテストのみ実行
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
        parser = argparse.ArgumentParser(description='YAMLありの例')
        parser.add_argument('config_path', type=str, help='設定ファイル(.yaml)')
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
    '''
    # メモ
    # model = hoge 
    # load_model = datapara (model)
    # lodal_model.load moge
    # model
    '''
    model = model.to(device)
    # save_model = model
    # model = nn.DataParallel(save_model)

    params = torch.load(weight_path, map_location=device)
    model.load_state_dict(params)

    test(config, model, dir_path)