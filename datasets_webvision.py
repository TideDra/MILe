from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights, ResNet18_Weights
import torch
from config import config
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import os
import json
from PIL import Image

class DataLoaderX(DataLoader):
    '''
    A replacement to DataLoader which improves the pipeline performance.
    '''

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class WebVision_Flickr(Dataset):
    # path是json文件路径,root是图片路径
    def __init__(self, root, path) -> None:
        super().__init__()
        self.label = []
        for json_name in os.listdir(path):
            data = json.load(open(path))
            for i in range(len(data)):
                self.label.append(data['tags'][i])
        self.data = {}
        self.img_list = os.listdir(root)
        self.img_list.sort()
        for file_list in img_list:
            for idx, file_name in enumerate(self.file_list):
                file_path = os.path.join(root, file_name)
                self.data[idx] = {
                    'image': file_path,
                    'label': self.label[idx]
                }

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    class CollateFn:
        def __init__(self, model) -> None:
            if model == 'resnet18':
                self.preprocess = ResNet18_Weights.DEFAULT.transforms()
            if model == 'resnet50':
                self.preprocess = ResNet50_Weights.DEFAULT.transforms()

        def __call__(self, batch):
            imgs = []
            labels = []
            for sample in batch:
                img = Image.open(sample['image']).convert('RGB')
                imgs.append(self.preprocess(img))
                labels.append(sample['label'])
            labels = torch.tensor(labels, dtype=torch.long)
            imgs = torch.stack(imgs)
            return imgs, labels

    def getDataloader(self, model):
        collate_fn = self.CollateFn(model)
        dataloader = DataLoaderX(self,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn,
                                 pin_memory=True,
                                 num_workers=config.num_workers)
        return dataloader

class WebvisionTrain:
    def __init__(self, root, model) -> None:
        if model == 'resnet18':
            self.preprocess = ResNet18_Weights.DEFAULT.transforms()
        if model == 'resnet50':
            self.preprocess = ResNet50_Weights.DEFAULT.transforms()
        self.dataset = Webvision_Flickr(root, split='train', transform=self.preprocess)

    def getDataloader(self):
        dataloader = DataLoaderX(self.dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=config.num_workers)
        return dataloader
