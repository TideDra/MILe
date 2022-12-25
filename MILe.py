import torch
import torchvision
from torchvision.models import resnet18,resnet50
from config import config
import tqdm
import torch.nn.functional as F


imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
dataloader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=False,
                                          num_workers=config.num_workers)

if config.model=='resnet50':
    teacher = resnet50()
    student = resnet50()
elif config.model=='resnet18':
    teacher = resnet18()
    student = resnet18()

teacher=teacher.to(config.device)
student=student.to(config.device)
if config.schema=='softmax':
    loss_fn=torch.nn.Softmax()
elif config.schema=='sigmoid' or config.schema=='MILe':
    weights=torch.tensor([0.001]*1000,device=config.device)
    loss_fn=torch.nn.BCEWithLogitsLoss(weight=weights,reduction='mean')

optimizer=torch.optim.SGD([{'params':teacher.parameters()},{'params':student.parameters()}],
                          momentum=0.9,
                          weight_decay=1e-4,
                          lr=config.lr)

lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                           mode='max',
                                           factor=0.1, 
                                           patience=5, 
                                           threshold=0.01, 
                                           threshold_mode='abs', 
                                           cooldown=0, 
                                           min_lr=1e-5, 
                                           eps=1e-08, 
                                           verbose=False)

############ Train ############


for epoch in range(config.epoch_num):
    t_loss=0
    with tqdm(enumerate(dataloader, start=1),
              unit='batch',
              total=config.k_t+config.k_s,
              desc='epoch:{}/{} interactive phase'.format(epoch, config.epochs),) as tbar:
        teacher.train()
        for batch_num,(input,label) in tbar:
            input.to(config.device)
            label.to(config.device)
            logits = teacher(input)
            loss = loss_fn(logits,label)
            t_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tbar.set_postfix(loss="%.4f" % (t_loss / batch_num))
            tbar.update()
            if batch_num==config.k_t:
                break
        
        tbar.set_description('epoch:{}/{} imitation phase'.format(epoch, config.epochs))
        teacher.eval()
        student.train()
        s_loss=0
        for batch_num,(input,_) in tbar:
            input.to(config.device)
            label = teacher(input)
            label = (label>config.rho).type(torch.float)
            logits = student(input)
            loss = loss_fn(logits,label)
            s_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tbar.set_postfix(loss="%.4f" % (s_loss / batch_num))
            tbar.update()
            if batch_num==config.k_s:
                break
        teacher.load_state_dict(student.state_dict())