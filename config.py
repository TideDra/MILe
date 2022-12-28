class config:
    model='resnet18'
    schema='softmax'
    data_path='/run/media/WinDoc/学习资料/大三上/ML/data/train/'
    val_path='/run/media/WinDoc/学习资料/大三上/ML/data/val'
    real_path='/run/media/WinDoc/学习资料/大三上/ML/data/real.json'
    mnist_path='/run/media/WinDoc/学习资料/大三上/ML/data/'
    dataset='imagenet'
    lr=0.1
    batch_size=64
    num_workers=4
    epoch_num=100
    k_t=800
    k_s=200
    device='cuda'
    rho=0.25
    checkpoint_path='./checkpoint'
    max_checkpoint_num=2