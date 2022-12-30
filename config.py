class config:
    model='resnet18'
    schema='MILe'
    data_path='../data/train/'
    val_path='../data/val'
    real_path='../data/real.json'
    mnist_path='../data/'
    dataset='imagenet'
    lr=0.1
    batch_size=16
    num_workers=4
    epoch_num=100
    k_t=8000
    k_s=2000
    device='cuda'
    rho=0.25
    checkpoint_path='./checkpoint'
    max_checkpoint_num=2