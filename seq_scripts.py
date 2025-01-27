import torch
import numpy as np
from tqdm import tqdm
ttl_dict = dict()
std_dict = dict()

def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    for batch_idx, batch_data in enumerate(tqdm(loader)):
        ret_dict = model(batch_data, device, epoch_idx=epoch_idx)
        loss = ret_dict['loss']
        loss_details = ret_dict['loss_details']
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('nan loss')
            continue
        optimizer.zero_grad()
        loss.backward()
        loss_value.append(loss.item())
        optimizer.step()
        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                f'Epoch: {epoch_idx}, Batch({batch_idx}/{len(loader)}) done. Loss: {loss.item():.4f}  lr:{clr[0]:.6f}'
            )
            recoder.print_log(
                ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_details.items()])
            )
    optimizer.scheduler.step()
    recoder.print_log(f'\tMean training loss: {np.mean(loss_value):.4f}.')
    return loss_value

def update_evaluation_info(evaluation_info, batch_data, ret_dict):
    if 'label' not in evaluation_info:
        evaluation_info['label'] = []
        evaluation_info['pred'] = []
        evaluation_info['fid'] = []
    evaluation_info['label'] += batch_data['label'].tolist()
    evaluation_info['pred'] += ret_dict['logits'].argsort(
        dim=1, descending=True)[:, :10].tolist()
    evaluation_info['fid'] += batch_data['fname']

def cal_acc(ret_info, recoder, epoch, mode):
    acc_result = np.array(ret_info['label'])[:, None] == (np.array(ret_info['pred']))
    for i in [1]:
        recoder.print_log(
            f"Epoch {epoch}, {mode} Topk-{i}\t: {acc_result[:, :i].sum()/len(acc_result)*100:.2f}%"
        )
    return acc_result[:, :1].sum()/len(acc_result) * 100

def seq_eval(
        cfg, loader, model, device, mode, epoch, work_dir, recoder, evaluate_tool="python"):

    model.eval()
    evaluation_info = dict()
    for batch_idx, batch_data in enumerate(tqdm(loader)):
        with torch.no_grad():
            ret_dict = model(batch_data, device, dataset=mode)
        
        batch_data['fname'] = [item['fid'] for item in batch_data['info']]
        update_evaluation_info(evaluation_info, batch_data, ret_dict)
    return cal_acc(evaluation_info, recoder, epoch, mode)

def update_ttldict(ret_info,view):
    for i in range(len(ret_info['fid'])):
        fid = ret_info['fid'][i]
        pred = ret_info['pred'][i][0]
        label = ret_info['label'][i]
        ttl_dict[fid + '_' + view] = pred
        std_dict[fid + '_' + view] = label

def gen_final_res(gloss_dict):
    writef = open('./answer.txt','w')
    stdwritef = open('./std_answer.txt','w')
    readf = open("./TEST_VIDEO_NAME.txt",'r')
    vidorder = readf.read().splitlines()
    for v in vidorder:
        pred = gloss_dict['id2gloss'][str(ttl_dict[v])]
        label = gloss_dict['id2gloss'][str(std_dict[v])]
        writef.write(pred + '\n')   
        stdwritef.write(label + '\n')          

def seq_final(
        cfg,view, loader, model, device, mode, epoch, work_dir, recoder, evaluate_tool="python"):

    model.eval()
    evaluation_info = dict()
    for batch_idx, batch_data in enumerate(tqdm(loader)):
        with torch.no_grad():
            ret_dict = model(batch_data, device, dataset=mode)
        
        batch_data['fname'] = [item['fid'] for item in batch_data['info']]
        update_evaluation_info(evaluation_info, batch_data, ret_dict)
        
    update_ttldict(evaluation_info,view)

