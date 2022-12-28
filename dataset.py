from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights,ResNet18_Weights
import torch
from config import config
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
import os
import json
from PIL import Image
class DataLoaderX(DataLoader):
    '''
    A replacement to DataLoader which improves the pipeline performance.
    '''
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ImageNetReal(Dataset):
    def __init__(self,root,real_path,origin_path) -> None:
        super().__init__()
        self.real_labels = json.load(open(real_path))
        self.origin_labels = open(origin_path).readlines()
        self.data={}
        self.file_list = os.listdir(root)
        self.file_list.sort()
        for idx,file_name in enumerate(self.file_list):
            file_path = os.path.join(root,file_name)
            self.data[idx]={
                'image':file_path,
                'original_label':int(self.origin_labels[idx].replace('\n','')),
                'real_label':self.real_labels[idx]
            }

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
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
                img=Image.open(sample['image']).convert('RGB')
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
        self.dataset = ImageNet(root,split='train',transform=self.preprocess)
    
    def getDataloader(self):
        dataloader = DataLoaderX(self.dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers)
        return dataloader

