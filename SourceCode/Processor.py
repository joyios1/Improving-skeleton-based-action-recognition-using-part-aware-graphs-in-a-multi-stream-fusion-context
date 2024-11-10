from __future__ import print_function
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
import numpy as np
import glob
import csv
import pylab

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import tsne_torch
from torch.utils.data import DataLoader

import yaml
from tqdm import tqdm

from tensorboardX import SummaryWriter
from SourceCode.LabelSmoothingCrossEntropy import LabelSmoothingCrossEntropy
from SourceCode.graph.ntu_rgb_d_classes import ntu_classes


def import_class(import_str):
    # class_srt = 'Model', mod_str = 'model.ctrgcn', _sep = '.'
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Processor:
    """
        Processor for Skeleton-based Action Recognition
    """

    def __init__(self, arg):
        self.arg = arg
        init_seed(self.arg.seed)
        self.save_arg()
        if arg.phase == 'train':
            arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
            # if os.path.isdir(arg.model_saved_name):
            #     print('log_dir: ', arg.model_saved_name, 'already exist')
            #     answer = input('delete it? y/n:')
            #     if answer == 'y':
            #         shutil.rmtree(arg.model_saved_name)
            #         print('Directory removed: ', arg.model_saved_name)
            #         input('Refresh the website of tensorboard by pressing any keys')
            #     else:
            #         print('Directory not removed: ', arg.model_saved_name)
            self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
            self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')

        # region Variables initialization
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.current_time = 0
        self.input_weights = None
        self.global_step = 0
        self.best_val_accuracy = -np.Inf
        self.best_val_acc_epoch = 0
        self.early_stop_cnt = 0
        self.should_early_stop = False
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.monitor_operation = np.greater_equal
        self.columns_number = 70
        # endregion

        self.loss = nn.CrossEntropyLoss().cuda(self.output_device)
        # self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(self.output_device)
        self.load_model()
        self.load_optimizer()
        self.load_data()

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=self.output_device)

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.arg.base_lr, momentum=0.9, nesterov=self.arg.nesterov, weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        if self.arg.phase == 'train' and self.arg.weights is None:
            self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def load_data(self):
        feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = DataLoader(
                dataset=feeder(**self.arg.train_feeder_args), batch_size=self.arg.batch_size,
                shuffle=True, num_workers=self.arg.num_worker, drop_last=True, worker_init_fn=init_seed)
        self.data_loader['val'] = DataLoader(
            dataset=feeder(**self.arg.val_feeder_args), batch_size=self.arg.batch_size,
            shuffle=False, num_workers=self.arg.num_worker, drop_last=False, worker_init_fn=init_seed)
        self.data_loader['test'] = DataLoader(
                dataset=feeder(**self.arg.test_feeder_args), batch_size=self.arg.test_batch_size,
                shuffle=False, num_workers=self.arg.num_worker, drop_last=False, worker_init_fn=init_seed)

    def load_model(self):
        model = import_class(self.arg.model)
        if self.arg.phase == 'train':
            shutil.copy2(inspect.getfile(model), self.arg.work_dir)
        self.model = model(**self.arg.model_args)

        if self.arg.weights:
            # self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.arg.start_epoch = int(self.arg.weights[:-3].split('-')[-2])
            self.print_log('Loading weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(self.output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

            self.print_log(f"Resuming training from epoch: {self.arg.start_epoch}")

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            previous_lr = self.lr
            if epoch < self.arg.warm_up_epoch:
                self.lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                self.lr = self.arg.base_lr * (self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

            if abs(previous_lr - self.lr) > 0.00001:
                self.print_log("--------- New learning rate is :  {:.5f}   ------------".format(self.lr))
            return self.lr
        else:
            raise ValueError()

    def adjust_learning_rate_cos_ann(self, epoch, idx):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            previous_lr = self.lr
            if epoch <= self.arg.warm_up_epoch:
                self.lr = self.arg.base_lr * epoch / self.arg.warm_up_epoch
            else:
                # lr = self.arg.base_lr * (
                #         self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
                T_max = len(self.data_loader['train']) * (self.arg.num_epoch - self.arg.warm_up_epoch)
                T_cur = len(self.data_loader['train']) * (epoch - 1 - self.arg.warm_up_epoch) + idx

                eta_min = self.arg.base_lr * self.arg.lr_ratio
                self.lr = eta_min + 0.5 * (self.arg.base_lr - eta_min) * (1 + np.cos((T_cur / T_max) * np.pi))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
            # if abs(previous_lr - self.lr) > 0.0001:
            return self.lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str_to_print, print_time=True, new_line=False):
        if new_line:
            print('\n')
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print('\n', file=f)
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str_to_print = "[ " + localtime + ' ] ' + str_to_print
        print(str_to_print)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str_to_print, file=f)

    def split_time(self):
        split_time = time.time() - self.current_time
        self.current_time = time.time()
        return split_time

    def get_number_of_parameters(self, is_trainable=False):
        if is_trainable:
            return sum(p.numel() for p in self.model.parameters())
        else:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def prepare_run(self, mode):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        self.print_log(mode.title() + 'ing...')
        self.current_time = time.time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        results = dict(batch_accuracies=list(), batch_scores=list(), batch_losses=list())
        process = tqdm(self.data_loader[mode], ncols=self.columns_number)
        return process, timer, results

    def update_tensorboard_params(self, epoch, accuracy, loss):
        self.train_writer.add_scalar('accuracy', accuracy*100, epoch)
        self.train_writer.add_scalar('loss', loss, epoch)
        self.train_writer.add_scalar('learning rate', self.lr, epoch)

    def train(self, epoch, save_model=False):
        process, timer, results = self.prepare_run('train')
        for batch_idx, (data, label, index) in enumerate(process):
            self.adjust_learning_rate_cos_ann(epoch, batch_idx)
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # forward
            output = self.model(data)
            loss = self.loss(output, label)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            timer['model'] += self.split_time()

            # statistics
            results['batch_losses'].append(loss.data.item())
            value, predict_label = torch.max(output.data, 1)
            batch_accuracy = torch.mean((predict_label == label.data).float())
            results['batch_accuracies'].append(batch_accuracy.data.item())
            self.lr = self.optimizer.param_groups[0]['lr']
            timer['statistics'] += self.split_time()

        mean_accuracy = np.mean(results['batch_accuracies'])
        mean_loss = np.mean(results['batch_losses'])

        # statistics of time consumption and loss
        proportion = {k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))for k, v in timer.items()}
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}, [Statistics]{statistics}'.format(**proportion))
        self.print_log('\tMean training loss: {:.3f}.  Mean training accuracy: {:.3f}%.'.format(mean_loss, mean_accuracy*100))
        self.print_log("\t--------- Current learning rate is:  {:.5f}   ------------".format(self.lr))

        self.update_tensorboard_params(epoch, mean_accuracy*100, loss)

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')

    def validate(self, epoch, save_model=True):
        process, timer, results = self.prepare_run('val')
        for batch_idx, (data, label, index) in enumerate(process):
            # graphWriter = SummaryWriter("runs/val")
            # graphWriter.add_graph(self.model.to('cpu'), data)
            # graphWriter.close()
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)

                output = self.model(data)
                loss = self.loss(output, label)
                results['batch_losses'].append(loss.data.item())
                results['batch_scores'].append(output.data.cpu().numpy())

        val_loss = np.mean(results['batch_losses'])
        val_accuracy = self.data_loader['val'].dataset.top_k(np.concatenate(results['batch_scores']), 1)

        self.update_tensorboard_params(epoch, val_accuracy * 100, loss)

        self.print_log('\tMean validation loss: {:.3f}. Top-1 validation accuracy: {:.3f}%'.format(val_loss, val_accuracy*100))

        current_val_accuracy = val_accuracy
        if self.monitor_operation(current_val_accuracy, self.best_val_accuracy):
            previous_best_val_accuracy = self.best_val_accuracy
            self.best_val_accuracy = current_val_accuracy
            self.best_val_acc_epoch = epoch
            self.early_stop_cnt = 0
            self.print_log('\t--------Validation accuracy improved from {:.3f}% to {:.3f}%--------'
                           .format(previous_best_val_accuracy*100, self.best_val_accuracy*100))

        else:
            self.early_stop_cnt += 1

        if self.early_stop_cnt > 20:
            self.print_log('\t--------Epoch %d: early stopping--------' % epoch)
            self.should_early_stop = True

        if self.arg.num_epoch == epoch:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')

    def test(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        # fs = list()
        # ls = list()

        for ln in loader_name:
            process, timer, results = self.prepare_run(ln)
            # sum_sigmas = torch.zeros(128).cuda(self.output_device)
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                    # output, l, f = self.model(data, label)
                    output = self.model(data)
                    loss = self.loss(output, label)

                    results['batch_scores'].append(output.data.cpu().numpy())
                    results['batch_losses'].append(loss.data.item())
                    value, predict_label = torch.max(output.data, 1)

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(results['batch_scores'])
            mean_loss = np.mean(results['batch_losses'])

            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))

            self.best_acc = self.data_loader[ln].dataset.top_k(score, 1)
            #self.best_acc_epoch = epoch

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {:.3f}.'.format(ln, len(self.data_loader[ln]), mean_loss))
            for k in self.arg.show_top_k:
                self.print_log('\tTop{}: {:.3f}%'.format(k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

    def start(self):
        if self.arg.phase == 'train':
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            # self.print_log(f'# All Parameters: {self.get_number_of_parameters(False)}')
            self.print_log(f'# Trainable Parameters: {self.get_number_of_parameters(True)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                e = epoch + 1
                self.print_log(f"EPOCH {e}:", new_line=True)
                save_model = ((e % self.arg.save_interval == 0) or (e == self.arg.num_epoch)) and e > self.arg.save_epoch
                self.train(e, save_model=save_model)
                self.validate(e, save_model=save_model)
                if self.should_early_stop:
                    break

            self.print_log('Best %s: %.3f from epoch:%d' % ('validation accuracy', self.best_val_accuracy*100, self.best_val_acc_epoch), new_line=True)

            # test the best model
            self.print_log('Testing best validation accuracy epoch : ' + str(self.best_val_acc_epoch))
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_val_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')

            self.arg.print_log = True
            self.test(epoch=0, save_score=True, loader_name=['val', 'test'], wrong_file=wf, result_file=rf)

            # test final epoch
            #self.print_log('Testing final epoch : ' + str(self.arg.num_epoch), new_line=True)
            #weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.arg.num_epoch)+'*'))[0]
            #weights = torch.load(weights_path)
            #if type(self.arg.device) is list:
            #    if len(self.arg.device) > 1:
            #        weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            #self.model.load_state_dict(weights)

            #wf = weights_path.replace('.pt', '_wrong.txt')
            #rf = weights_path.replace('.pt', '_right.txt')

            #self.arg.print_log = True
            #self.test(epoch=0, save_score=True, loader_name=['val', 'test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Testing accuracy: {self.best_acc * 100}%')
            #self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Best validation accuracy: {self.best_val_accuracy * 100}%')
            self.print_log(f'Epoch number: {self.best_val_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = True
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.test(epoch=0, save_score=self.arg.save_score, loader_name=['test', 'val'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')
