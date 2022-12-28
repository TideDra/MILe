from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights,ResNet18_Weights
import torch
from config import config
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder,MNIST
import os
import json
from PIL import Image
from random import random
import numpy as np
class DataLoaderX(DataLoader):
    '''
    A replacement to DataLoader which improves the pipeline performance.
    '''
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class ImageNetReal(ImageFolder):
    def __init__(self, root: str,real_path):
        super().__init__(root)
        self.real_labels = json.load(open(real_path))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        id = int(os.path.splitext(path)[0].split('_')[-1])-1
        return {'image':sample,
                'original_label':target,
                'real_label':self.real_labels[id]
                }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    class CollateFn:
        def __init__(self,model) -> None:
            if model=='resnet18':
                self.preprocess = ResNet18_Weights.DEFAULT.transforms()
            if model=='resnet50':
                self.preprocess = ResNet50_Weights.DEFAULT.transforms()
            
        def __call__(self,batch):
            imgs=[]
            labels=[]
            real_labels=[]
            for sample in batch:
                img=sample['image']
                imgs.append(self.preprocess(img))
                labels.append(sample['original_label'])
                real_labels.append(sample['real_label'])
            labels=torch.tensor(labels,dtype=torch.long)
            imgs=torch.stack(imgs)
            return imgs,labels,real_labels
    
    def getDataloader(self,model):
        collate_fn=self.CollateFn(model)
        dataloader = DataLoaderX(self,
                            batch_size=config.batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            pin_memory=True,
                            num_workers=config.num_workers)
        return dataloader

class ImageNetTrain:
    def __init__(self,root,model) -> None:
        if model=='resnet18':
            self.preprocess = ResNet18_Weights.DEFAULT.transforms()
        if model=='resnet50':
            self.preprocess = ResNet50_Weights.DEFAULT.transforms()
        self.dataset = ImageFolder(root,transform=self.preprocess)
    
    def getDataloader(self):
        dataloader = DataLoaderX(self.dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=config.num_workers)
        return dataloader

class MultilabelMNIST:
    def __init__(self,root,model) -> None:
        self.train_set = MNIST(root,train=True,download=True)
        self.test_set = MNIST(root,train=False,download=True)
        self.model=model
    
    class CollateFn:
        def __init__(self,model,train) -> None:
            if model=='resnet18-10':
                self.preprocess = ResNet18_Weights.DEFAULT.transforms()
            self.train=train
        def __call__(self, batch):
            imgs=[]
            single_labels=[]
            real_labels=[]
            for i in range(len(batch)):
                batch[i]=list(batch[i])
                batch[i][0]=torch.tensor(np.array(batch[i][0].convert('RGB')))
            for i in range(0,len(batch),9):
                if i+8>=len(batch):
                    break
                up = torch.concatenate([batch[i][0],batch[i+1][0],batch[i+2][0]],dim=1)
                mid = torch.concatenate([batch[i+3][0],batch[i+4][0],batch[i+5][0]],dim=1)
                down = torch.concatenate([batch[i+6][0],batch[i+7][0],batch[i+8][0]],dim=1)
                combine = torch.concatenate([up,mid,down],dim=0)
                combine = combine.permute(2,0,1)
                combine = self.preprocess(combine)
                imgs.append(combine)
                p=random()
                if p<=0.6:
                    label=batch[i+4][1]
                elif p<=0.65:
                    label=batch[i][1]
                elif p<=0.7:
                    label=batch[i+1][1]
                elif p<=0.75:
                    label=batch[i+2][1]
                elif p<=0.8:
                    label=batch[i+3][1]
                elif p<=0.85:
                    label=batch[i+5][1]
                elif p<=0.9:
                    label=batch[i+6][1]
                elif p<=0.95:
                    label=batch[i+7][1]
                else:
                    label=batch[i+8][1]
                single_labels.append(label)
                real_label=[batch[i+j][1] for j in range(8)]
                real_label=set(real_label)
                real_label=list(real_label)
                real_labels.append(real_label)
            single_labels=torch.tensor(single_labels)
            imgs=torch.stack(imgs)
            if self.train:
                return imgs,single_labels
            else:
                return imgs,single_labels,real_labels
        
    def getDataloader(self,train):
        if train:
            collatefn=self.CollateFn(self.model,train)
            dataloader=DataLoader(self.train_set,
                            batch_size=config.batch_size*9,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=collatefn,
                            num_workers=config.num_workers)
        else:
            collatefn=self.CollateFn(self.model,train)
            dataloader=DataLoader(self.test_set,
                            batch_size=config.batch_size*9,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=collatefn,
                            num_workers=config.num_workers)
        return dataloader

                