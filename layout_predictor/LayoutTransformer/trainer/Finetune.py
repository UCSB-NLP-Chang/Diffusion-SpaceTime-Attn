import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import random
import torch.nn as nn
import math
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
from .scheduler import Scheduler
import pickle


class FinetuneTrainer:    
    def __init__(self, model, dataloader, opt):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.opt = opt
        self.n_epochs = opt.n_epochs
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.device = self._prepare_gpu()
        self.tb_writer = SummaryWriter(log_dir='saved' + "/finetune/tensorboard/")
        self.model = model
        self.pad_index = 0
        self.bos_index = 1
        self.eos_index = 2
        self.mask_index = 3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4, betas=(0.9, 0.999), weight_decay=0.01)
        self.scheduler = Scheduler(self.optimizer, self.model.hidden_size, n_warmup_steps=10000)
        self.loss = nn.NLLLoss(ignore_index=self.pad_index, reduction='sum')
        self.begin_epoch = 0
        self.all_log = []
        self.noise_dim = 64
        self._resume_checkpoint(opt.checkpoint)
        with open('data/rel_dict.pkl', 'rb') as file:
            self.vocab_dict = pickle.load(file)

    def train(self):
        opt = self.opt
        all_log = self.all_log
        self.model.to(self.device)
        self.logger.info('[STRUCTURE]')
        self.logger.info(self.model)
        for i in range(self.begin_epoch, self.begin_epoch + self.n_epochs):
            log = self._run_epoch(i, 'train')
            val_log = self._run_epoch(i, 'valid')
            merged_log = {**log, **val_log}
            all_log.append(merged_log)
            if (i + 1)%5 == 0:
                checkpoint = {
                    'log': all_log,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'n_steps': self.scheduler.n_current_steps,
                }

                check_path = os.path.join(opt.save_dir, 'checkpoint_' + str(i+1) + '.pth')
                torch.save(checkpoint, check_path)
                print("SAVING CHECKPOINT:", check_path)

    def test(self):
        self.model.to(self.device)
        self._run_epoch(self.begin_epoch, 'test')


    def _run_epoch(self, epoch, phase):
        if phase == 'train':
            self.model.train()
            dataloader = self.dataloader
        else:
            self.model.eval()
            dataloader = self.dataloader.split_validation()

        total_loss = 0
        total_vocab_loss = 0
        total_token_type_loss = 0

        total_correct = 0
        total_label = 0

        for batch_idx, (input_token, output_label, segment_label, token_type)  in enumerate(dataloader):

            input_token = input_token.to(self.device)
            output_label = output_label.to(self.device)
            segment_label = segment_label.to(self.device)
            token_type = token_type.to(self.device)
            input_noise = torch.randn(input_token.size()[0], input_token.size()[1], self.noise_dim).to(self.device)
            src_mask = (input_token != 0).unsqueeze(1).to(self.device)

            if phase == 'train':
                vocab_logits, token_type_logits = self.model(input_token, segment_label, token_type, src_mask, self.noise_dim)
            else:
                vocab_logits, token_type_logits = self.model(input_token, segment_label, token_type, src_mask, self.noise_dim)

            # compute log probs
            log_probs_vocab = F.log_softmax(vocab_logits, dim=-1)
            log_probs_type = F.log_softmax(token_type_logits, dim=-1)
            # print("log_probs_cats shape:", log_probs_cats.size())

            # NLLLoss: Src-> N*C (C for classes), Trg-> N 
            log_probs_vocab = log_probs_vocab.reshape(log_probs_vocab.size(0) * log_probs_vocab.size(1), log_probs_vocab.size(2))
            log_probs_type = log_probs_type.reshape(log_probs_type.size(0) * log_probs_type.size(1), log_probs_type.size(2))
            trg_vocab = output_label.reshape(output_label.size(0) * output_label.size(1))
            trg_type = token_type.reshape(token_type.size(0) * token_type.size(1))

            # compute batch loss
            vocab_loss = self.loss(log_probs_vocab, trg_vocab)
            type_loss = self.loss(log_probs_type, trg_type)
            # print("Loss shape:", log_probs_cats.size(), trg_cats_id.size())

            loss = (vocab_loss + type_loss) / (2 * input_token.size(0))

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.scheduler.step_and_update_lr()

            correct, total = self._calc_acc(log_probs_vocab, trg_vocab)
            total_correct += correct
            total_label += total

            if phase == 'test':
                print("INPUT:", self.idx2vocab(input_token[0, :16].detach().cpu().numpy()))
                print("GT:", self.idx2vocab(output_label[0, :16].detach().cpu().numpy()))
                print("PRED:", self.idx2vocab(torch.max(vocab_logits[0, :16], dim=1)[1].detach().cpu().numpy()))

            total_loss += loss.item()
            total_vocab_loss += vocab_loss.item()
            total_token_type_loss += type_loss.item()
            print('[%d/%d] Loss: %.4f Loss_vocab: %.4f Loss_token_type: %.4f'% \
                (batch_idx + 1, len(dataloader), loss.item(), vocab_loss.item(), type_loss.item()))
            
        if phase == 'train':
            acc = (total_correct.float() / total_label.float()).item()
            log = self._log_epoch(epoch, total_loss, total_vocab_loss, total_token_type_loss, acc, 'train')
        else:
            acc = (total_correct.float() / total_label.float()).item()
            log = self._log_epoch(epoch, total_loss, total_vocab_loss, total_token_type_loss, acc, 'valid')

            # Find testcase with [MASK] and log
            for sent_num in range(len(input_token)):
                if 3 in input_token[sent_num]:
                    log["INPUT"] = self.idx2vocab(input_token[sent_num, :16].detach().cpu().numpy())
                    log["GT"] = self.idx2vocab(output_label[sent_num, :16].detach().cpu().numpy())
                    log["PRED"] = self.idx2vocab(torch.max(vocab_logits[sent_num, :16], dim=1)[1].detach().cpu().numpy())
                    break

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
    
    def _log_epoch(self, epoch, total_loss, total_vocab_loss, total_token_type_loss, acc, phase):
        
        log = {
            'epoch': epoch,
            phase + '_loss': total_loss,
            phase + '_vocab_loss': total_vocab_loss,
            phase + '_token_type_loss': total_token_type_loss,
        }
        self.tb_writer.add_scalar( phase + "/Loss", total_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_vocab", total_vocab_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_token_type", total_token_type_loss, epoch)
        self.tb_writer.add_scalar( phase + "/mask_acc", acc, epoch)
        print("======================================================================================")
        print('FINISH EPOCH: [%d/%d] Loss: %.4f Loss_vocab: %.4f Loss_token_type: %.4f'%\
         (epoch + 1, self.n_epochs, total_loss, total_vocab_loss, total_token_type_loss))
        print("======================================================================================")
        print("FINISH EPOCH: [%d/%d] Mask word acc: %.4f"%(epoch + 1, self.n_epochs, acc))

        return log

    def idx2vocab(self, idx):
        sent = ""
        for i in range(len(idx)):
            sent += self.vocab_dict[idx[i]]
            sent += " "

        return sent


    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device

    def _resume_checkpoint(self, path):
        if path == None: return
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.resume_checkpoint(checkpoint['n_steps'])
            self.begin_epoch = checkpoint['log'][-1]['epoch'] + 1
            self.all_log = checkpoint['log']
        except:
            self.logger.error('[Resume] Cannot load from checkpoint')
