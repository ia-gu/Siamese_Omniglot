import torch
from torch.utils.data import Dataset
import os
from numpy.random import choice as npc
import numpy as np
import random
from PIL import Image
import csv

# dataloader for training
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
        with open('../../data/meta-dataset/csv/omniglot.csv', mode='r') as f:
            reader = csv.reader(f)
            dict_from_csv = {rows[2]:rows[3] for rows in reader}

        # load data
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

    def __len__(self):
        return 21000000

    def __getitem__(self, index):
        label = None

        # from the same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
            
        # from the different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


# dataloader for testing
class OmniglotTest(Dataset):

    def __init__(self, dataPath, transform=None, times=200, way=20):
        super(OmniglotTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)

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
        if idx == 0:
            label = 1.0
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
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