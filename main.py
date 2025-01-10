import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pdb
import sys

import json
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
from collections import OrderedDict

faulthandler.enable()
import utils
from seq_scripts import seq_train, seq_eval, generate_submission, results_analysis, seq_final, gen_final_res


class Processor():
    def __init__(self, arg):
        self.arg = arg

        self.accelerator = None
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.device.set_device(self.arg.device)

        self.save_arg()
        self.recoder = utils.Recorder(
            self.arg.work_dir, self.arg.print_log, self.arg.log_interval
            )
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = json.load(
            open(f'./data/gloss_dict.json', 'r')
            )
        self.model, self.optimizer = self.loading()

    def start(self):
        if self.arg.phase == 'train':
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            best_performance = 0.0
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                save_model = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                # train end2end model
                seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder)
                if eval_model:
                    # eval all test scene
                    performance = seq_eval(
                        self.arg, self.data_loader["val"], self.model, self.device,
                        "val", epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool
                        )
                    self.recoder.print_log(
                        f"Epoch {epoch}, Average Topk-{1}\t: {performance:.2f}%"
                    )
                if save_model:
                    if self.arg.num_epoch - epoch <= self.arg.keep_last:
                        model_path = f"{self.arg.work_dir}/model_{epoch}.pt"
                        self.save_model(epoch, model_path)
                    if performance > best_performance:
                        best_performance = performance
                        model_path = f"{self.arg.work_dir}/model_best.pt"
                        self.save_model(epoch, model_path)
        elif self.arg.phase == 'test':
            # if self.arg.load_weights is None and self.arg.load_checkpoints is None:
            #     raise ValueError('Please appoint --load-weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            seq_eval(
                self.arg, self.data_loader["val"], self.model, self.device,
                "val", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            # eval all test scene
            performance = 0
            for scene in ['ITW', 'STU', 'SYN', 'TED']:
            # for scene in ['TED']:
                for view in ['kl', 'kr']:
                    indictor = f'test_{scene}_{view}'
                    performance += seq_eval(
                        self.arg, self.data_loader[indictor], self.model, self.device,
                        indictor, 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool
                        )
            self.recoder.print_log(
                f"Epoch 6667, Average Topk-{1}\t: {performance/8:.2f}%"
            )
            self.recoder.print_log('Evaluation Done.\n')
        elif self.arg.phase == 'analysis':
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            results_analysis(self.data_loader, self.model, self.device, 'test', self.arg.work_dir, self.gloss_dict)
        elif self.arg.phase == "submission":
            for k, v in self.data_loader['test'].items():
                seq_eval(
                    self.arg, v, self.model, self.device,
                    f"test_{k}", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
        elif self.arg.phase == "final":
            ttl_dict = None
            for scene in ['ITW', 'STU', 'SYN', 'TED']:
            # for scene in ['TED']:
                for view in ['kl', 'kr']:
                    indictor = f'test_{scene}_{view}'
                    seq_final(
                        self.arg,view, self.data_loader[indictor], self.model, self.device,
                        indictor, 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            gen_final_res(self.gloss_dict)
            self.recoder.print_log('Evaluation Done.\n')

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def loading(self):
        self.load_data()
        self.recoder.print_info("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            class_num=len(self.gloss_dict['gloss2id']),
        )
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        
        model = self.model_to_device(model)
        self.recoder.print_info("Loading model finished.")
        return model, optimizer

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        return model
    
    def load_model_weights(self, model, weight_path):
        new_state_dict = model.state_dict()
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        state_dict = state_dict['model_state_dict']
        for key in state_dict.keys():
            if key in new_state_dict:
                new_state_dict[key] = state_dict[key]
        # 1. 记录所有成功加载的权重
        loaded_keys = set()
        
        # 2. 比较 state_dict 和 new_state_dict，找出匹配的权重
        for key in state_dict.keys():
            if key in new_state_dict:
                new_state_dict[key] = state_dict[key]
                if key == 'heads.classify.classifier.weight':
                    if self.arg.phase == 'train':
                        new_state_dict['heads.classify.classifier_s.weight']=state_dict[key]
                    if '2d_skeleton' in self.arg.feeder_args['train']['data_type'] and 'r_features' in self.arg.feeder_args['train']['data_type'] and not 'd_features' in self.arg.feeder_args['train']['data_type']: # RGB Fusion
                        if self.arg.phase == 'train':
                            if key == 'heads.classify.classifier.weight':
                                new_state_dict['heads.classify.classifier_r.weight'] = state_dict[key]
                        else: # the RGB-fusion model has different keys
                            new_state_dict['heads.classify.classifier_s.weight'] = state_dict['heads.classify.classifier2.weight']
                            new_state_dict['heads.classify.classifier_r.weight'] = state_dict['heads.classify.classifier3.weight']
                loaded_keys.add(key)
        
        # 3. 检查哪些权重没有加载成功
        not_loaded_keys = set(state_dict.keys()) - loaded_keys
        if not_loaded_keys:
            print("The following weights were NOT loaded:")
            for key in not_loaded_keys:
                print(f" - {key}")
        else:
            print("All weights loaded successfully.")

        model.load_state_dict(new_state_dict, strict=False)
    
    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        # state_dict = OrderedDict([(k.replace('visual_extractor', 'visual_backbone.gcn'), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints, map_location=torch.device('cpu'))

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device.output_device)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        self.recoder.print_info("Loading data")
        self.feeder = import_class(self.arg.feeder)
        dataset_list = zip(["train", "train_val", "val"], [True, False, False])
        kps_config = self.arg.feeder_args['kps_config']
        for idx, (mode, train_flag) in enumerate(dataset_list):
                arg = self.arg.feeder_args[mode.split('_')[0]]
                arg['mode'] = mode.split('_')[0]
                arg["kps_config"] = kps_config
                arg["transform_mode"] = train_flag
                self.dataset[mode] = self.feeder(
                        gloss_dict=self.gloss_dict,
                        osxposs=self.arg.aug_poss['osxposs'],
                        temporaltype=self.arg.model_args['temporal_arg']['type'],
                        **arg
                    )
                self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        self.recoder.print_info("Loading data finished.")
        self.load_test_data()
    
    def load_test_data(self):
        self.recoder.print_info("Loading test data")
        self.feeder = import_class(self.arg.feeder)
        for scene in ['ITW', 'STU', 'SYN', 'TED']:
            # for view in ['kf', 'kl', 'kr']:
            for view in ['kl', 'kr']:
                kps_config = self.arg.feeder_args['kps_config']
                mode, train_flag = 'test', False
                arg = self.arg.feeder_args[mode]
                arg['mode'] = mode.split('_')[0]
                arg["kps_config"] = kps_config
                arg["transform_mode"] = train_flag
                input_list_file = f'./data/{mode}_{scene}_self.json'
                indictor = f'test_{scene}_{view}'
                self.dataset[indictor] = self.feeder(
                        gloss_dict=self.gloss_dict, 
                        input_list_file=input_list_file,
                        view=view, # specify the view used
                        osxposs=self.arg.aug_poss['osxposs'],
                        temporaltype=self.arg.model_args['temporal_arg']['type'],
                        **arg
                    )
                self.data_loader[indictor] = self.build_dataloader(self.dataset[indictor], mode, train_flag)
        self.recoder.print_info("Loading data finished.")

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=lambda x:self.feeder.collate_fn(x,self.arg.model_args['temporal_arg']['type']),
        )


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    # p.config = "baseline_iter.yaml"
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    # with open(f"./configs/{args.dataset}.yaml", 'r') as f:
    #     args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    processor = Processor(args)
    processor.start()
