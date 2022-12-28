import torch
import torchvision
from torchvision.models import resnet18,resnet50
from config import config
from tqdm import tqdm
import torch.nn.functional as F
from dataset import ImageNetReal,ImageNetTrain
from utils import eval,save_model
from itertools import cycle
if __name__=='__main__':
    print('loading model...')
    if config.model=='resnet50':
        teacher = resnet50()
        student = resnet50()
    elif config.model=='resnet18':
        teacher = resnet18()
        student = resnet18()
    print('loading train dataset...')
    imagenet_train_data = ImageNetTrain(config.data_path,config.model)
    train_dataloader = imagenet_train_data.getDataloader()
    print('loading val dataset...')
    imagenet_val_data = ImageNetReal(config.val_path,config.real_path,config.origin_path)
    val_dataloader = imagenet_val_data.getDataloader(config.model)

    teacher=teacher.to(config.device)
    student=student.to(config.device)
    if config.schema=='softmax':
        loss_fn=torch.nn.CrossEntropyLoss(reduction='mean')
    elif config.schema=='sigmoid' or config.schema=='MILe':
        loss_fn=torch.nn.BCEWithLogitsLoss(reduction='mean')

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
    print('start training...')
    if config.schema=='MILe':
        for epoch in range(config.epoch_num):
            t_loss=0
            with tqdm(
                      unit='batch',
                      total=config.k_t+config.k_s,
                      desc='epoch:{}/{} interactive phase'.format(epoch, config.epoch_num),) as tbar:
                teacher.train()
                idx=0
                while True:
                    for batch_num,(input,label) in enumerate(train_dataloader,start=1):
                        input=input.to(config.device)
                        label=label.to(config.device)
                        logits = teacher(input)
                        label=F.one_hot(label,num_classes=logits.shape[-1])
                        label=label.float()
                        loss = loss_fn(logits,label)
                        t_loss+=loss.item()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        idx+=1
                        tbar.set_postfix(loss="%.4f" % (t_loss / idx))
                        tbar.update()
                        
                        if idx==config.k_t:
                            break
                    if idx==config.k_t:
                        break
                tbar.set_description('epoch:{}/{} imitation phase'.format(epoch, config.epoch_num))
                teacher.eval()
                student.train()
                s_loss=0
                idx=0
                while True:
                    for batch_num,(input,_) in enumerate(train_dataloader,start=1):
                        input=input.to(config.device)
                        label = teacher(input)
                        label = (label>config.rho).type(torch.float)
                        logits = student(input)
                        loss = loss_fn(logits,label)
                        s_loss+=loss.item()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        idx+=1
                        tbar.set_postfix(loss="%.4f" % (s_loss / idx))
                        tbar.update()
                        
                        if idx==config.k_s:
                            break
                    if idx==config.k_s:
                        break
                teacher.load_state_dict(student.state_dict())
                acc,real_f1,real_acc,label_coverage=eval(student,val_dataloader)
                with open('output.txt','a') as f:
                    f.write(f"epoch:{epoch} acc:{acc} real-acc:{real_acc} real-f1:{real_f1} label_coverage:{label_coverage}\n")
                lr_scheduler.step(real_f1)

                name=f"{config.schema}_{epoch}.pth"
                save_model(student,name)
    
    elif config.schema=='softmax' or config.schema=='sigmoid':
        for epoch in range(config.epoch_num):
            t_loss=0
            with tqdm(enumerate(train_dataloader, start=1),
                      unit='batch',
                      total=config.k_t+config.k_s,
                      desc='epoch:{}/{} interactive phase'.format(epoch, config.epoch_num),) as tbar:
                teacher.train()
                for batch_num,(input,label) in tbar:
                    input=input.to(config.device)
                    label=label.to(config.device)
                    logits = teacher(input)
                    if config.schema=='sigmoid':
                        label=F.one_hot(label,num_classes=logits.shape[-1])
                        label=label.float()
                    loss = loss_fn(logits,label)
                    t_loss+=loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tbar.set_postfix(loss="%.4f" % (t_loss / batch_num))
                    tbar.update()
                acc,real_f1,real_acc,label_coverage=eval(student,val_dataloader)
                lr_scheduler.step(real_f1)
                with open('output.txt','a') as f:
                    f.write(f"epoch:{epoch} acc:{acc} real-acc:{real_acc} real-f1:{real_f1} label_coverage:{label_coverage}\n")
                name=f"{config.schema}_{epoch}.pth"
                save_model(student,name)