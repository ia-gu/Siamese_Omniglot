import torch
from torch.utils.data import Dataset
import os
from numpy.random import choice as npc
import numpy as np
import random
from PIL import Image
import csv

# SiameseNet用のデータローダを自作
# 学習時
class OmniglotTrain(Dataset):

    def __init__(self, dataPath, train='train', transform=None):
        super(OmniglotTrain, self).__init__()
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath, train)

    def loadToMem(self, dataPath, train):
        print("begin loading training dataset to memory")
        datas = {}
        agrees = [0]
        idx = 0
        # csvを基準に，train/validで使用するデータを分ける分割
        with open('../../data/meta-dataset/csv/omniglot.csv', mode='r') as f:
            reader = csv.reader(f)
            dict_from_csv = {rows[2]:rows[3] for rows in reader}

        # 使用データ読み込み
        for agree in agrees:
            for alphaPath in os.listdir(dataPath):
                if(dict_from_csv[alphaPath+'-character01']==train):
                    for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                        datas[idx] = []
                        for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                            filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                            datas[idx].append(Image.open(filePath).rotate(agree).convert('L'))
                        idx += 1
        print("finish loading training dataset to memory")
        return datas, idx

    # データサイズ
    def __len__(self):
        return 21000000

    # データ，ラベルをセットで作成．
    def __getitem__(self, index):
        label = None

        # 同クラス
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
            
        # 他クラス
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        # 前処理
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


# テスト時
# 20クラス分類を行うために学習時とはちょっと違う実装
class OmniglotTest(Dataset):

    def __init__(self, dataPath, transform=None, times=200, way=20):
        super(OmniglotTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)

    # 学習時と同様
    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        data_num = 0
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                    data_num += 1
                idx += 1
        print(f'data_num:{data_num}, class_num:{idx}')
        print("finish loading test dataset to memory")
        return datas, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # 同クラスの場合(ミニバッチの先頭)
        if idx == 0:
            label = 1.0
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
        # 他クラスの場合(先頭以外)
        else:
            label = 0.0
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])
        
        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))