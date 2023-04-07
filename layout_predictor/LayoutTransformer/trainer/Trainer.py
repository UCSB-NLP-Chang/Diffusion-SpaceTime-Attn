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
from .scheduler import Scheduler
import pickle


class Trainer:    
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
        self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr = 1e-5, betas=(0.9, 0.999))
        self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr = 1e-4, betas=(0.9, 0.999))
        self.encoder_scheduler = Scheduler(self.encoder_optimizer, self.model.encoder.hidden_size, n_warmup_steps=10000)
        self.decoder_scheduler = Scheduler(self.decoder_optimizer, self.model.decoder.hidden_size, n_warmup_steps=10000)
        self.loss = nn.NLLLoss(ignore_index=self.pad_index, reduction='sum')
        self.begin_epoch = 0
        self.all_log = []
        with open('data/rel_dict.pkl', 'rb') as file:
            self.vocab_dict = pickle.load(file)
        with open('data/cls_dict.pkl', 'rb') as file:
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
        total_pos_loss = 0
        total_shape_loss = 0
        
        total_correct = [0, 0, 0]
        total_label = [0, 0, 0]

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
                trg_input_pos = center_pos[:, :-1]
                trg_input_shape = shape_centroid[:, :-1]
                trg_mask = (trg_input_cats_id != self.pad_index).unsqueeze(1).to(self.device)
                # print("trg embedding shape:", trg_input.size())

            trg_cats_id = cats_id[:, 1:]
            trg_pos = center_pos[:, 1:]
            trg_shape = shape_centroid[:, 1:]
            
#             trg_input_cats_id = cats_id[:, :]
#             trg_mask = (trg_input_cats_id != self.pad_index)
#             forcing_p = random.random()
#             if forcing_p < 1 or phase != 'train':
#                 trg_input_pos = ((torch.ones(trg_input_cats_id.size()).long() * 3).to(self.device) & trg_mask.long())
#                 trg_input_shape = ((torch.ones(trg_input_cats_id.size()).long() * 3).to(self.device) & trg_mask.long())
#             else:
#                 trg_input_pos = center_pos[:, :]
#                 trg_input_shape = shape_centroid[:, :]
#             trg_mask = trg_mask.unsqueeze(1)

#             trg_cats_id = cats_id[:, :]
#             trg_pos = center_pos[:, :]
#             trg_shape = shape_centroid[:, :]
            # print("trg shape:", trg_cats_id.size())

            if phase == 'train':
                output_cats, output_pos, output_shape = self.model(
                input_token, input_id, segment_label, token_type, src_mask, trg_input_cats_id, trg_input_pos, trg_input_shape, trg_mask)
            else:
                output_cats, output_pos, output_shape, pred_cats, pred_pos, pred_shape = \
                self.model.inference(
                input_token, input_id, segment_label, token_type, src_mask)

#             print('output:',output_cats[:,-1].shape)
            # compute log probs
            log_probs_cats = F.log_softmax(output_cats, dim=-1)
            log_probs_pos = F.log_softmax(output_pos, dim=-1)
            log_probs_shape = F.log_softmax(output_shape, dim=-1)
            # print("log_probs_cats shape:", log_probs_cats.size())

            # NLLLoss: Src-> N*C (C for classes), Trg-> N 
            log_probs_cats = log_probs_cats.reshape(log_probs_cats.size(0) * log_probs_cats.size(1), log_probs_cats.size(2))
            log_probs_pos = log_probs_pos.reshape(log_probs_pos.size(0) * log_probs_pos.size(1), log_probs_pos.size(2))
            log_probs_shape = log_probs_shape.reshape(log_probs_shape.size(0) * log_probs_shape.size(1), log_probs_shape.size(2))
            trg_cats_id = trg_cats_id.reshape(trg_cats_id.size(0) * trg_cats_id.size(1))
            trg_pos = trg_pos.reshape(trg_pos.size(0) * trg_pos.size(1))
            trg_shape = trg_shape.reshape(trg_shape.size(0) * trg_shape.size(1))

            # compute batch loss
            cats_loss = self.loss(log_probs_cats, trg_cats_id) / cats_id.size(0)
            pos_loss = self.loss(log_probs_pos, trg_pos) / cats_id.size(0)
            shape_loss = self.loss(log_probs_shape, trg_shape) / cats_id.size(0)
            # print("Loss shape:", log_probs_cats.size(), trg_cats_id.size())


#             loss = (cats_loss*0.9 + pos_loss*0.05 + shape_loss*0.05)
            loss = (cats_loss*0.4 + pos_loss*0.3 + shape_loss*0.3)
#             loss = (cats_loss*1. + pos_loss*0. + shape_loss*0.)

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            if phase == 'train':
                loss.backward()
                self.encoder_scheduler.step_and_update_lr()
                self.decoder_scheduler.step_and_update_lr()

            
            correct, total = self._calc_acc(log_probs_cats, trg_cats_id)
            total_correct[0] += correct
            total_label[0] += total
            correct, total = self._calc_acc(log_probs_pos, trg_pos)
            total_correct[1] += correct
            total_label[1] += total
            correct, total = self._calc_acc(log_probs_shape, trg_shape)
            total_correct[2] += correct
            total_label[2] += total

            total_loss += loss.item()
            total_cat_loss += cats_loss.item()
            total_pos_loss += pos_loss.item()
            total_shape_loss += shape_loss.item()

            if batch_idx % (len(dataloader)//4) == 0:
                print('[%d/%d] Loss: %.4f Loss_cat: %.4f Loss_pos: %.4f Loss_shape: %.4f'% \
                    (batch_idx + 1, len(dataloader), loss.item(), cats_loss.item(), pos_loss.item(), shape_loss.item()))
                
        print("INPUT:", self.idx2vocab(input_token[0, :16].detach().cpu().numpy(), 'text'))
        print("GT:", self.idx2vocab(cats_id[0, 1:17].detach().cpu().numpy(), 'img'))
        print("OUTPUT:", self.idx2vocab(torch.max(output_cats[0, :16], dim=1)[1].detach().cpu().numpy(), 'img'))
            

        if phase == 'train':
            acc = [(total_correct[0].float() / total_label[0].float()).item(), 
                   (total_correct[1].float() / total_label[1].float()).item(), 
                   (total_correct[2].float() / total_label[2].float()).item()]
            log = self._log_epoch(epoch, total_loss/len(dataloader), 
                                  total_cat_loss/len(dataloader), 
                                  total_pos_loss/len(dataloader), 
                                  total_shape_loss/len(dataloader), acc, 'train', 
                                  self.decoder_optimizer)
        else:
            acc = [(total_correct[0].float() / total_label[0].float()).item(), 
                   (total_correct[1].float() / total_label[1].float()).item(), 
                   (total_correct[2].float() / total_label[2].float()).item()]
            log = self._log_epoch(epoch, total_loss/len(dataloader),
                                  total_cat_loss/len(dataloader),
                                  total_pos_loss/len(dataloader),
                                  total_shape_loss/len(dataloader), acc, 'valid', 
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
    
    def _log_epoch(self, epoch, total_loss, total_cat_loss, total_pos_loss, total_shape_loss, \
        acc, phase, optimizer):
        
        log = {
            'epoch': epoch,
            phase + '_loss': total_loss,
            phase + '_cat_loss': total_cat_loss,
            phase + '_pos_loss': total_pos_loss,
            phase + '_shape_loss': total_shape_loss,
        }
        self.tb_writer.add_scalar( phase + "/Loss", total_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_cat", total_cat_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_pos", total_pos_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_shape", total_shape_loss, epoch)
        self.tb_writer.add_scalar( phase + "/lr", optimizer.param_groups[0]['lr'], epoch)

        self.tb_writer.add_scalar( phase + "/obj_acc", acc[0], epoch)
        self.tb_writer.add_scalar( phase + "/pos_acc", acc[1], epoch)
        self.tb_writer.add_scalar( phase + "/shp_acc", acc[2], epoch)
        print("="*30)
        print('FINISH EPOCH: [%d/%d] Loss: %.4f Loss_cat: %.4f Loss_pos: %.4f Loss_shape: %.4f'%\
         (epoch + 1, self.n_epochs, total_loss, total_cat_loss, total_pos_loss, total_shape_loss))
        print("="*30)
        print("FINISH EPOCH: [%d/%d] PRED OBJ acc: %.4f"%(epoch + 1, self.n_epochs, acc[0]))
        return log

    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device

    def _resume_checkpoint(self, path):
        if path == None: 
            checkpoint = torch.load('saved/pretrained_v6/checkpoint_50.pth')

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
