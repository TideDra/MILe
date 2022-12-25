class config:
    num_workers=16
    model='restnet50'
    schema='MILe'
    lr=0.1
    batch_size=256
    epoch_num=100
    k_t=8000
    k_s=2000
    device='cuda'
    rho=0.25