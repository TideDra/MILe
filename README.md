# MILe
An unofficial implementation of MILe in [Multi-label Iterated Learning for Image Classification with Label Ambiguity](https://openaccess.thecvf.com/content/CVPR2022/html/Rajeswar_Multi-Label_Iterated_Learning_for_Image_Classification_With_Label_Ambiguity_CVPR_2022_paper.html).

## Features
### Backbones
This implementation supports the following models as the backbone of MILe
- ResNet18 for 1000 classes
- ResNet50 for 1000 classes
- ResNet18 for 10 classes

Note that it's easy for you to use other backbones by replacing `teacher` and `student` with the backbone in the code. But by that, you may need to modify the code of data preprocessing.

### Benchmarks
I have implemented some benchmarks used in the paper.
- ImageNet1k (for training) and ImageNet Real (for validating)
- Multi-label MNIST

## Get Started
Run the following command to train MILe.
```
python MILe.py
```

## Configuration
You can modify all the configurations in `config.py`
- **model**: the backbone of MILe. Valid options:
    - `resnet18`
    - `resnet50`
    - `resnet18-10`
- **schema**: training strategy used in the paper. Valid options:
    - `softmax` softmax + cross entropy loss
    - `sigmoid` sigmoid + BCE loss
    - `MILe` sigmoid + BCE loss + iterated learning
- **data_path**: path of the ImageNet train set. The structure of the directory should be:
  ```
  train/
    -- n01440764/
    -- n01443537/
    ...
  ```
- **val_path**: path of the ImageNet val set. The structure of the directory should be the same as the train set.
- **real_path**: path of the Real label file. See [here](https://github.com/google-research/reassessed-imagenet/blob/master/real.json).
- **mnist_path**: path of the MNIST dataset. The structure of the directory should be:
  ```
  MNIST/
    -- raw/
      -- t10k-images-idx3-ubyte
      ...
  ```
  You don't need to manually download it because the `torchvision.dataset.MNIST` will do that if there is no dataset. So just set this parameter the root directory.

- **dataset**: the benchmark you use. Valid options:
    -- `imagenet`, then the model should be `resnet18` or `resnet50`
    -- `mnist`, then the model should be `resnet18-10`

- **lr**: learning rate
- **batch_size**
- **num_workers**: num_worker of dataloader
- **epoch_num**: number of training epochs.
- **k_t**: iteration number of interactive phase.
- **k_s**: iteration number of imitation phase
- **device**: training device. Valid options:
    -- `cuda`
    -- `cpu`
- **rho**: threshold of pseudo-label
- **checkpoint_path**: saving path of checkpoint
- **max_checkpoint_num**: maximum number of checkpoints in the saving path.