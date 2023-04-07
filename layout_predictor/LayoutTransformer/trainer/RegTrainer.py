import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import random
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from .scheduler import Scheduler, ChrisScheduler
from .loss import RegLoss
import pickle


class RegTrainer:    
    def __init__(self, model, dataloader, opt):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.opt = opt
        self.n_epochs = opt.n_epochs
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.device = self._prepare_gpu()
        self.tb_writer = SummaryWriter(log_dir=os.path.join(opt.save_dir, 'tensorboard'))
        self.model = model
        self.model.to(self.device)
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr = opt.lr, betas=(0.9, 0.999))
        self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr = opt.lr, betas=(0.9, 0.999))
        self.encoder_scheduler = ChrisScheduler(self.encoder_optimizer, 
                    self.model.encoder.hidden_size, n_warmup_steps=10000, 
                    n_hold_steps=100, n_decay_steps=70000, max_lr=opt.lr/20., min_lr=opt.lr/100.)
        self.decoder_scheduler = ChrisScheduler(self.decoder_optimizer, 
                    self.model.decoder.hidden_size, n_warmup_steps=10000, 
                    n_hold_steps=100, n_decay_steps=70000, max_lr=opt.lr, min_lr=opt.lr/100.)
        
        for p in self.model.encoder.parameters():
            p.requires_grad = False
            
        self.NLLLoss = nn.NLLLoss(ignore_index=self.pad_index, reduction='sum')
        self.RegLoss = RegLoss(reduction = 'mean')
        self.begin_epoch = 0
        self.all_log = []
        
        with open('data/coco/object_pred_idx_to_name.pkl', 'rb') as file:
            self.vocab_dict = pickle.load(file)
        with open('data/coco/object_pred_idx_to_name.pkl', 'rb') as file:
            self.cls_dict = pickle.load(file)

        self._resume_checkpoint(opt.checkpoint)


    def train(self):
        opt = self.opt
        all_log = self.all_log
        # self.model.to(self.device)
        self.logger.info('[STRUCTURE]')
        self.logger.info(self.model)
        for i in range(self.begin_epoch, self.begin_epoch + self.n_epochs):
            log = self._run_epoch(i, 'train')
            if (i + 1)%1 == 0:
                val_log = self._run_epoch(i, 'valid')
                merged_log = {**log, **val_log}
                all_log.append(merged_log)
            else:
                all_log.append(log)
            if (i + 1)%5 == 0:
                checkpoint = {
                    'log': all_log,
                    'state_dict': self.model.state_dict(),
                    'encoder_optimizer': self.encoder_optimizer.state_dict(),
                    'decoder_optimizer': self.decoder_optimizer.state_dict(),
                    'encoder_n_steps': self.encoder_scheduler.n_current_steps,
                    'decoder_n_steps': self.decoder_scheduler.n_current_steps,
                }

                check_path = os.path.join(opt.save_dir, 'checkpoint_' + str(i+1) + '.pth')
                torch.save(checkpoint, check_path)
                print("SAVING CHECKPOINT:", check_path)

    def test(self):
        # self.model.to(self.device)
        self._run_epoch(self.begin_epoch, 'test')

    def _run_epoch(self, epoch, phase):
        if phase == 'train':
            self.model.train()
            dataloader = self.dataloader
        else:
            self.model.eval()
            dataloader = self.dataloader.split_validation()

        total_loss = 0
        total_cat_loss = 0
        total_box_loss = 0
        
        total_correct = 0
        total_label = 0

        for batch_idx, (input_token, segment_label, token_type, \
            cats_id, center_pos, shape_centroid, boxes_xy, input_id)  in \
            enumerate(dataloader):

            cats_id = cats_id.to(self.device)
            input_id = input_id.to(self.device)
            center_pos = center_pos.to(self.device)
            shape_centroid = shape_centroid.to(self.device)
            boxes_xy = boxes_xy.to(self.device)
            input_token = input_token.to(self.device)
            segment_label = segment_label.to(self.device)
            token_type = token_type.to(self.device)
            src_mask = (input_token != 0).unsqueeze(1).to(self.device)

            # if phase == 'train':
            if phase == 'train':
                trg_input_cats_id = cats_id[:, :-1]
                trg_input_box = boxes_xy[:, :-1]
                trg_mask = (trg_input_cats_id != self.pad_index).unsqueeze(1).to(self.device)
                # print("trg embedding shape:", trg_input.size())
            
            trg_input_template = cats_id[:, 1:]

            trg_cats_id = cats_id[:, 1:]
            trg_box = boxes_xy[:, 1:]

            if phase == 'train':
                output_cats, output_box = self.model(
                    input_token, input_id, segment_label, token_type, src_mask,
                    trg_input_cats_id, trg_input_box, trg_mask, trg_input_template)
            else:
                output_cats, output_box, pred_cats, pred_box = self.model.inference(
                    input_token, input_id, segment_label, token_type, src_mask,
                    trg_input_template)

#             print('output:',output_cats[:,-1].shape) 
            # compute log probs 
            log_probs_cats = F.log_softmax(output_cats, dim=-1) 
            # print("log_probs_cats shape:", log_probs_cats.size()) 
            # NLLLoss: Src-> N*C (C for classes), Trg-> N 

            log_probs_cats = log_probs_cats.reshape(
                log_probs_cats.size(0) * log_probs_cats.size(1), log_probs_cats.size(2))
            trg_cats_id = trg_cats_id.reshape(trg_cats_id.size(0) * trg_cats_id.size(1))
            output_box = output_box.reshape(
                output_box.size(0) * output_box.size(1), output_box.size(2))
            trg_box = trg_box.reshape(trg_box.size(0) * trg_box.size(1), trg_box.size(2))
            
            # compute batch loss
            cats_loss = self.NLLLoss(log_probs_cats, trg_cats_id) / cats_id.size(0)
            boxes_loss = self.RegLoss(output_box, trg_box) # cats_id.size(0)
            # print("Loss shape:", log_probs_cats.size(), trg_cats_id.size())

#             loss = (cats_loss*0.9 + pos_loss*0.05 + shape_loss*0.05)
            self.l = 5.
            loss = cats_loss + self.l * (boxes_loss) 
#             loss = (cats_loss*1. + pos_loss*0. + shape_loss*0.)

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            if phase == 'train':
                loss.backward()
                self.encoder_scheduler.step_and_update_lr()
                self.decoder_scheduler.step_and_update_lr()

            
            correct, total = self._calc_acc(log_probs_cats, trg_cats_id)
            total_correct += correct
            total_label += total

            total_loss += loss.item()
            total_cat_loss += cats_loss.item()
            total_box_loss += (self.l * boxes_loss).item()

            if batch_idx % (len(dataloader)//4) == 0:
                print('[%d/%d] Loss: %.4f Loss_cat: %.4f Loss_box: %.4f'% \
                    (batch_idx + 1, len(dataloader), loss.item(), cats_loss.item(),
                    (self.l * boxes_loss).item()))
                
        print("INPUT:", self.idx2vocab(input_token[0, :16].detach().cpu().numpy(), 'text'))
        print("GT:", self.idx2vocab(cats_id[0, 1:17].detach().cpu().numpy(), 'img'))
        print("OUTPUT:", self.idx2vocab(torch.max(output_cats[0, :16], dim=1)[1].detach().cpu().numpy(), 'img'))
            
        if phase == 'train':
            acc = (total_correct.float() / total_label.float()).item()
            log = self._log_epoch(epoch, total_loss/len(dataloader), 
                                  total_cat_loss/len(dataloader), 
                                  total_box_loss/len(dataloader), acc, 'train', 
                                  self.decoder_optimizer)
        else:
            acc = (total_correct.float() / total_label.float()).item()
            log = self._log_epoch(epoch, total_loss/len(dataloader),
                                  total_cat_loss/len(dataloader),
                                  total_box_loss/len(dataloader), acc, 'valid', 
                                  self.decoder_optimizer)

        return log

    def _calc_acc(self, logits, gt):
        """
        Param
            logits: Tensor, (B * max_length, C)
            gt:   Tensor, (B * max_length)
        """
        pred = torch.max(logits, dim=1)[1]
        correct = torch.sum((pred==gt) & (gt != 0))
        total = torch.sum((gt != 0))
        return correct, total

    def idx2vocab(self, idx, modality):
        sent = ""
        for i in range(len(idx)):
            if modality == 'text':
                sent += self.vocab_dict[idx[i]]
            else:
                sent += self.cls_dict[idx[i]]
            sent += " "

        return sent
    
    def _log_epoch(self, epoch, total_loss, total_cat_loss, total_box_loss,
                   acc, phase, optimizer):
        
        log = {
            'epoch': epoch,
            phase + '_loss': total_loss,
            phase + '_cat_loss': total_cat_loss,
            phase + '_box_loss': total_box_loss,
        }
        self.tb_writer.add_scalar( phase + "/Loss", total_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_cat", total_cat_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_box", total_box_loss, epoch)
        self.tb_writer.add_scalar( phase + "/lr", optimizer.param_groups[0]['lr'], epoch)

        self.tb_writer.add_scalar( phase + "/obj_acc", acc, epoch)
        print("="*30)
        print('FINISH EPOCH: [%d/%d] Loss: %.4f Loss_cat: %.4f Loss_box: %.4f '%\
         (epoch + 1, self.n_epochs, total_loss, total_cat_loss, total_box_loss,))
        print("="*30)
        print("FINISH EPOCH: [%d/%d] PRED OBJ acc: %.4f"%(epoch + 1, self.n_epochs, acc))
        return log

    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device

    def _resume_checkpoint(self, path):
        if path == None: 
            checkpoint = torch.load('saved/pretrained_coco_v5/checkpoint_50.pth')

            self.model.encoder.load_state_dict(checkpoint['state_dict'])
            self.encoder_optimizer.load_state_dict(checkpoint['optimizer'])
            self.encoder_scheduler.resume_checkpoint(checkpoint['n_steps'])
            return
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
            self.encoder_scheduler.resume_checkpoint(checkpoint['encoder_n_steps'])
            self.decoder_scheduler.resume_checkpoint(checkpoint['decoder_n_steps'])
            self.begin_epoch = checkpoint['log'][-1]['epoch'] + 1
            self.all_log = checkpoint['log']
        except:
            self.logger.error('[Resume] Cannot load from checkpoint')
