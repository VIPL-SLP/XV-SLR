import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import chain
ttl_dict = dict()
std_dict = dict()

def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    loss_value = []
    # nclass = model.num_classes
    # confusion_matrix = np.zeros((nclass, nclass), dtype=int)
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
        # for i in range(len(ret_dict['logits'])):
        #     confusion_matrix[batch_data['label'][i].item(), ret_dict['logits'].argmax(dim=-1)[i].item()] += 1 
        if batch_idx % recoder.log_interval == 0:
            # acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix) * 100
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
    # for i in [1, 5, 10]:
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

def recoder_results(evaluation_info, names, labels, preds):
    for (name, label, pred) in zip(names, labels, preds):
        if name in evaluation_info.keys():
            continue
        evaluation_info[name] = {}
        evaluation_info[name]['label'] = label
        evaluation_info[name]['pred'] = pred

def ACCv2(ret_info):
    all_label = [v['label'] for k, v in ret_info.items()]
    all_preds = [v['pred'] for k, v in ret_info.items()]
    acc_result = np.array(all_label)[:, None] == (np.array(all_preds))
    ret_info['acc'] = [round(acc_result[:, :i].sum()/len(acc_result) * 100, 6) for i in [1, 5, 10]]

def dump_results(info, work_dir, scene, gloss_dict):
    from tabulate import tabulate
    file = open(os.path.join(work_dir, f'test_{scene}.txt'), 'w')
    file.write(f'Results of Test-{scene}\n')
    
    # # print acc first
    # data = [
    #     ['TOP1'] + [info[k]['acc'][0] for k in ['kf', 'kl', 'kr']],
    #     ['TOP5'] + [info[k]['acc'][1] for k in ['kf', 'kl', 'kr']],
    #     ['TOP10'] + [info[k]['acc'][2] for k in ['kf', 'kl', 'kr']]
    # ]
    # header = ['Acc.', 'kf', 'kl', 'kr']
    # table = tabulate(data, header, tablefmt="psql")
    # file.write(table)
    # file.write('\n\n-------------------------------------------------------------------------\n\n')

    # print all predictions
    header = ['file id', 'label', 'kf', 'kl', 'kr']
    data = []
    for k, v in info['kf'].items():
        if k == 'acc':
            continue
        label = gloss_dict['id2gloss'][str(v['label'])]
        temp = [k, label]
        for x in ['kf', 'kl', 'kr']:
            pred = gloss_dict['id2gloss'][str(info[x][k]['pred'][0])]
            if not pred == label:
                temp.append('|' + pred + '|')
            else:
                temp.append(pred)
        data.append(temp)
    table = tabulate(data, header, tablefmt="psql")
    file.write(table)
    file.close()

def overall_view(all_preds, work_dir, gloss_dict):
    from tabulate import tabulate
    file = open(os.path.join(work_dir, f'detail_analysis.txt'), 'w')
    header = ['Acc', ' ', 'ITW', ' ', ' ', 'STU', ' ',' ' , 'SYN', ' ', ' ', 'TED', ' ']
    data = [[' '] + ['kf', 'kl', 'kr'] * 4]
    for i in range(3):
        if i == 0:
            all_acc = ['TOP1']
        elif i == 1:
            all_acc = ['TOP5']
        else:
            all_acc = ['TOP10']
        for scene in ['ITW', 'STU', 'SYN', 'TED']:
            all_acc.extend([all_preds[scene][k]['acc'][i] for k in ['kf', 'kl', 'kr']])
        data.append(all_acc)
    table = tabulate(data, header, tablefmt="psql")
    file.write(table)

    file.write('\n\n-------------------------------------------------------------------------\n\n')

    header = ['label', ' ', 'ITW', ' ', ' ', 'STU', ' ',' ' , 'SYN', ' ', ' ', 'TED', ' ']
    data = [[' '] + ['kf', 'kl', 'kr'] * 4]
    
    gloss_count = {}
    for scene in ['ITW', 'STU', 'SYN', 'TED']:
        for view in ['kf', 'kl', 'kr']:
            for k, v in all_preds[scene][view].items():
                if k == 'acc':
                    continue
                label = gloss_dict['id2gloss'][str(v['label'])]
                pred = gloss_dict['id2gloss'][str(v['pred'][0])]
                if not label in gloss_count:
                    gloss_count[label] = {
                        'ITW': {'kf': [], 'kl': [], 'kr': []},
                        'STU': {'kf': [], 'kl': [], 'kr': []},
                        'SYN': {'kf': [], 'kl': [], 'kr': []},
                        'TED': {'kf': [], 'kl': [], 'kr': []}
                    }
                if not pred == label:
                    gloss_count[label][scene][view].append(k)
    for k, v in gloss_count.items():
        temp = [[k], [' ']]
        for scene in ['ITW', 'STU', 'SYN', 'TED']:
            for view in ['kf', 'kl', 'kr']:
                for i in range(2):
                    if i < len(gloss_count[k][scene][view]):
                        temp[i].append(gloss_count[k][scene][view][i])
                    else:
                        temp[i].append(' ')
                temp[0][1] += '\n'
        temp.append(['-'] * len(temp[0]))
        data.extend(temp)
    table = tabulate(data, header, tablefmt="psql")
    file.write(table)

def results_analysis(
        all_loader, model, device, mode, work_dir, gloss_dict, **kwargs
        ):
    
    def gather_object(obj):
        output = [None for _ in range(accelerator.num_processes)]
        torch.distributed.all_gather_object(output, obj)
        output = list(chain(*output))
        return output

    model.eval()
    merge_preds = {}
    for scene in ['ITW', 'STU', 'SYN', 'TED']:
        all_preds = {}
        for view in ['kf', 'kl', 'kr']:
            indictor = f'test_{scene}_{view}'
            loader = all_loader[indictor]
            all_preds[view] = {}
            for batch_idx, batch_data in enumerate(tqdm(loader)):
                with torch.no_grad():
                    ret_dict = model(batch_data, device,dataset=mode)
                    
                fname = [item['fid'] for item in batch_data['info']]
                labels = ret_dict['label'].cpu().tolist()
                preds = ret_dict['logits'].argsort(dim=1, descending=True)[:, :10].tolist()
                recoder_results(all_preds[view], fname, labels, preds)
            ACCv2(all_preds[view])
        dump_results(all_preds, work_dir, scene, gloss_dict)
        merge_preds[scene] = all_preds
    overall_view(merge_preds, work_dir, gloss_dict)


def generate_submission(loader, model, device, mode, work_dir, recoder, test_views):
    model.eval()
    total_results = {}
    # run test for different Scenario
    for k, cur_loader in loader.items():
        total_results[k] = {}
        for batch_idx, batch_data in tqdm(enumerate(cur_loader)):
            fid = [item[0] for item in batch_data['info']]
            for view in ['kf', 'kl', 'kr']:
                pass

def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))
