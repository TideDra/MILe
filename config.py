class config:
    num_workers=16
    model='resnet18'
    schema='MILe'
    data_path='/run/media/WinDoc/学习资料/大三上/ML/data/'
    val_path='/run/media/WinDoc/学习资料/大三上/ML/data/ILSVRC2012_img_val'
    real_path='/run/media/WinDoc/学习资料/大三上/ML/data/real.json'
    origin_path='/run/media/WinDoc/学习资料/大三上/ML/data/ILSVRC2012_validation_ground_truth.txt'
    lr=0.1
    batch_size=256
    num_workers=4
    epoch_num=100
    k_t=8000
    k_s=2000
    device='cuda'
    rho=0.25
    checkpoint_path='./checkpoint'
    max_checkpoint_num=2