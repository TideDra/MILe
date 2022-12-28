from config import config
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score
def save_model(model, name):
    print('Saving checkpoint...\n')

    model_name = model.__class__.__name__
    checkpoint_path = config.checkpoint_path
    if model_name not in os.listdir(checkpoint_path):
        os.mkdir(os.path.join(checkpoint_path, model_name))
    checkpoint_path = os.path.join(checkpoint_path, model_name)
    torch.save(model.state_dict(),
                     os.path.join(checkpoint_path, name))
    checkpoint_list = os.listdir(checkpoint_path)
    if (len(checkpoint_list) > config.max_checkpoint_num):
        file_map = {}
        times = []
        del_num = len(checkpoint_list) - config.max_checkpoint_num
        for f in checkpoint_list:
            t = f.split('.')[0].split('_')[-1]
            file_map[int(t)] = os.path.join(checkpoint_path, f)
            times.append(int(t))
        times.sort()
        for i in range(del_num):
            del_f = file_map[times[i]]
            os.remove(del_f)
    print('Checkpoint has been updated successfully.\n')

def eval(model,imgNet_dataloader):
    model.eval()
    with tqdm(enumerate(imgNet_dataloader),
              unit='batch',
              total=len(imgNet_dataloader),
              desc='evaluating origin val set') as tbar:
        sample_num=0
        hard_correct_num=0
        soft_correct_num=0
        total_real_f1=0
        total_f1=0
        total_coverage=0
        f1_num=0
        with torch.no_grad():
            for batch_num,(input,label,real_label) in tbar:
                input=input.to(config.device)
                label=label.to(config.device)
                logits = model(input)
                if config.schema=='softmax':
                    prob = F.softmax(logits,dim=1)
                if config.schema=='sigmoid' or config.schema=='MILe':
                    prob = torch.sigmoid(logits)
                hard_pred = torch.argmax(prob,dim=1)

                hard_correct_num += (hard_pred == label).sum().item()
                hard_pred=hard_pred.tolist()
                for idx,p in enumerate(hard_pred):
                    if len(real_label[idx]) and p in real_label[idx]:
                        soft_correct_num+=1

                raw_soft_pred = (prob>config.rho).nonzero().tolist()
                soft_pred = [[] for i in range(len(real_label))]
                for p in raw_soft_pred:
                    soft_pred[p[0]].append(p[1])         

                for i in range(len(real_label)):
                    tp,fp=0,0
                    pred_set = soft_pred[i]
                    real_set = real_label[i]
                    for p in pred_set:
                        if p in real_set:
                            tp+=1
                        else:
                            fp+=1
                    fn = len(real_set)-tp
                    if 2*tp+fp+fn!=0:
                        total_real_f1 += 2*tp/(2*tp+fp+fn)
                    try:
                        total_f1+=2*(tp/(tp+fp)*(tp/len(real_set)))/((tp/(tp+fp))+tp/len(real_set))
                    except:
                        pass
                    if len(real_set):
                        total_coverage+=len(pred_set)/len(real_set)
                    f1_num+=1
                sample_num+=len(label)
                tbar.update()

        real_f1=round(total_real_f1/sample_num,3)
        f1=round(total_f1/sample_num,3)
        real_acc=round(soft_correct_num/sample_num,3)
        acc=round(hard_correct_num/sample_num,3)
        label_coverage=round(total_coverage/f1_num,3)
        print(f"acc:{acc} real-acc:{real_acc} real-f1:{real_f1} f1:{f1} label_coverage:{label_coverage} ")
        return acc,real_f1,real_acc,label_coverage