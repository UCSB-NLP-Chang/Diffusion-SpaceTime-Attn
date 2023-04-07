import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import torch.nn as nn
from .loss import RegLoss, FocalLoss, Log_Pdf, Rel_Loss, Customized_Gmm_Loss, Customized_Box_Loss, Customized_Hinge_Loss

import matplotlib.pyplot as plt
from .scheduler import build_scheduler
import pickle
from .iou import IOU_calculator


class PretrainTrainer:    
    def __init__(self, model, dataloader, dataloader_r, opt, cfg):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.opt = opt
        self.cfg = cfg
        self.n_epochs = self.cfg['SOLVER']['EPOCHS']
        self.save_dir = self.cfg['OUTPUT']['OUTPUT_DIR']
        self.two_path = self.cfg['MODEL']['DECODER']['TWO_PATH']
        self.dataloader = dataloader
        self.dataloader_r = dataloader_r
        self.batch_size = dataloader.batch_size
        self.total_steps = len(dataloader) * self.n_epochs
        self.device = self._prepare_gpu()
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'tensorboard'))
        self.model = model
        self.pad_index = 0
        self.bos_index = 1
        self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), 
                                          betas=(0.9, 0.999), weight_decay=0.01)
        self.bbox_head_optimizer = torch.optim.Adam(self.model.bbox_head.parameters(), 
                                          betas=(0.9, 0.999), weight_decay=0.01)
#         self.scheduler = MultiStepLR(self.optimizer, milestones=[4, 30,40,60], 
# gamma=0.6667)
#         self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.6667, patience=5,
#                                            threshold = 0.02, threshold_mode='rel')
        self.encoder_scheduler = build_scheduler(cfg, self.encoder_optimizer, self.total_steps, "ENC")
        self.bbox_head_scheduler = build_scheduler(cfg, self.bbox_head_optimizer, self.total_steps, "BOX")
        self.loss = nn.NLLLoss(ignore_index=self.pad_index, reduction='sum')
        # self.focal_loss = FocalLoss(gamma=2, alpha=None, ignore_index=self.pad_index, reduction='sum')
        self.IOU_c = IOU_calculator(reduction = 'mean', cfg=cfg)
        self.val_box_loss = RegLoss(reduction='sum',pretrain = True, lambda_xy = 1., lambda_wh = 1., refine=True)
        self.refine, self.box_loss, self.rel_loss, self.refine_box_loss, self.customized_gmm_loss, \
            self.customized_bbox_loss, self.customized_hinge_loss = self.build_loss()

        
        if self.cfg['MODEL']['ENCODER']['ENABLE_NOISE']:
            self.noise_size = self.cfg['MODEL']['ENCODER']['NOISE_SIZE']
            
        self.begin_epoch = 0
        self.all_log = []
        self._resume_checkpoint(opt.checkpoint)
        self.pretrain_encoder = cfg['MODEL']['PRETRAIN']
        if not self.pretrain_encoder:
            assert self.cfg['SOLVER']['PRETRAIN_WEIGHT'] is not '', 'Please input the pretrain checkpoint.'
            self._load_encoder_weight(self.cfg['SOLVER']['PRETRAIN_WEIGHT'])
            self.pretrain_encoder = False

        # TODO:
        # This will be fix in the future
        if self.cfg['DATASETS']['NAME'] == 'coco':
            with open(os.path.join(self.cfg['DATASETS']['DATA_DIR_PATH'], 
                                   'object_pred_idx_to_name.pkl'), 'rb') as file:
                self.vocab_dict = pickle.load(file)
        elif self.cfg['DATASETS']['NAME'] == 'vg_msdn':
            with open(os.path.join(self.cfg['DATASETS']['DATA_DIR_PATH'], 
                                   'object_pred_idx_to_name.pkl'), 'rb') as file:
                self.vocab_dict = pickle.load(file)
        elif self.cfg['DATASETS']['NAME'] == 'vg_co':
            with open(os.path.join(self.cfg['DATASETS']['DATA_DIR_PATH'], 
                                   'object_pred_idx_to_name.pkl'), 'rb') as file:
                self.vocab_dict = pickle.load(file)
        elif self.cfg['DATASETS']['NAME'] == 'vg':
            with open(os.path.join(self.cfg['DATASETS']['DATA_DIR_PATH'], 
                                self.cfg['DATASETS']['REL_DICT_FILENAME']), 'rb') as file:
                self.vocab_dict = pickle.load(file)
            with open(os.path.join(self.cfg['DATASETS']['DATA_DIR_PATH'], 
                                self.cfg['DATASETS']['CLS_DICT_FILENAME']), 'rb') as file:
                self.cls_dict = pickle.load(file)

                
    def train(self):
        opt = self.opt
        all_log = self.all_log
        self.model.to(self.device)
        best_val_mIOU = 0.
        for i in range(self.begin_epoch, self.begin_epoch + self.n_epochs):
            if self.two_path and i%2 == 0:
                mode = 'w'
            elif self.two_path and i%2 == 1:
                mode = 'r'
            else: mode = 'w'
            log = self._run_epoch(i, 'train', mode)
            val_log = self._run_epoch(i, 'valid', 'w')
            # merged_log = {**log, **val_log}
            # all_log.append(merged_log)
            if (i + 1)%10 == 0 or val_log['valid_coarse_miou'] > best_val_mIOU:
                if val_log['valid_coarse_miou'] > best_val_mIOU:
                    best_val_mIOU = val_log['valid_coarse_miou']
                checkpoint = {
                    'log': None,
                    'state_dict': self.model.state_dict(),
                    'encoder_optimizer': self.encoder_optimizer.state_dict(),
                    'bbox_head_optimizer': self.bbox_head_optimizer.state_dict(),
                    'n_steps': self.encoder_scheduler.n_current_steps,
                }

                check_path = os.path.join(self.save_dir, 'checkpoint0219_' + str(i+1) + '_{}'.format(val_log['valid_coarse_miou']) + '.pth')
                torch.save(checkpoint, check_path)
                self.logger.info("SAVING CHECKPOINT: {}".format(check_path))

    def test(self):
        self.model.to(self.device)
        self._run_epoch(self.begin_epoch, 'test')

    def _run_epoch(self, epoch, phase, mode):
        # This is the key code for training.
        # many losses are declared in parent repo, but we only have two loss (absolute, relative)
        self.logger.info('[Phase: {}, Epoch: {}]'.format(phase, epoch))
        if phase == 'train':
            self.model.train()
            if mode == 'w':
                dataloader = self.dataloader
            else:
                dataloader = self.dataloader_r
        else:
            self.model.eval()
            if mode == 'w':
                dataloader = self.dataloader.split_validation()
            else:
                dataloader = self.dataloader_r.split_validation()
        total_loss = 0
        total_rel_box_loss = 0
        total_coar_box_loss = 0

        coarse_miou = 0
        refine_miou = 0

        for batch_idx, inputs in enumerate(dataloader):
            bpe_toks = inputs['bpe_toks']
            object_index = inputs['object_index']
            caption = inputs['caption']
            relation = inputs['relation']
            src_masks = []
            for each in bpe_toks:
                src_mask = (each != 1).to(each)
                src_masks.append(src_mask)
            bpe_toks_tensor = torch.Tensor(len(bpe_toks), bpe_toks[0].shape[0]).int()
            src_masks_tensor = torch.Tensor(len(bpe_toks), bpe_toks[0].shape[0]).int()
            torch.cat(bpe_toks, out=bpe_toks_tensor)
            torch.cat(src_masks, out=src_masks_tensor)
            bpe_toks_tensor = bpe_toks_tensor.to("cuda")
            src_masks_tensor = src_masks_tensor.unsqueeze(1).to(bpe_toks_tensor)

            trg_tmp = bpe_toks_tensor[:,:-1]
            trg_mask = (trg_tmp != 0).unsqueeze(1).to(self.device)
            trg_mask[:,0] = 1

            object_tensors = []
            for each_object in object_index:
                object_tensor = torch.zeros(128).to(torch.bool)
                curr_object_index = each_object
                for each_curr_object_index in curr_object_index:
                    object_tensor[each_curr_object_index] = True
                object_tensors.append(object_tensor)
            object_tensor = torch.stack(object_tensors).to("cuda")

            bpe_label_pair = None
            if phase == 'train':
                coarse_box, coarse_gmm, refine_box, refine_gmm = self.model(bpe_toks_tensor, src_masks_tensor, bpe_label_pair, epoch=epoch, trg_mask=trg_mask, object_pos_tensor=object_tensor)
            else:
                coarse_box, coarse_gmm, refine_box, refine_gmm = self.model(bpe_toks_tensor, src_masks_tensor, bpe_label_pair, inference = True, epoch=epoch, trg_mask=trg_mask, object_pos_tensor=object_tensor)
                
            real_loss = torch.tensor(0.).cuda()
            gmm_loss  = torch.tensor(0.).cuda()
            above_loss = torch.tensor(0.).cuda()
            below_loss = torch.tensor(0.).cuda()
            left_loss = torch.tensor(0.).cuda()
            right_loss = torch.tensor(0.).cuda()
            cnts = [0,0,0,0]

            coarse_box_output_for_loss = []
            coarse_gmm_output_for_loss = []
            relation_for_loss = []
            caption_for_loss = []
            full_relation_for_loss = []

            gt_xy_gmm_loss = []
            generate_xy_gmm_loss = []
            for instance_idx in range(coarse_box.shape[0]):
                curr_relation = relation[instance_idx]
                if len(curr_relation) == 0:
                    continue
                if len(curr_relation[0]) == 3:
                    # process with hinge loss
                    for _, each_relation in enumerate(curr_relation):
                        coarse_box_output_for_loss.append(coarse_box[instance_idx][each_relation[0]])
                        coarse_box_output_for_loss.append(coarse_box[instance_idx][each_relation[1]])
                        coarse_gmm_output_for_loss.append(coarse_gmm[instance_idx][each_relation[0]])
                        coarse_gmm_output_for_loss.append(coarse_gmm[instance_idx][each_relation[1]])
                        relation_for_loss.append(each_relation[2])
                        full_relation_for_loss.append(each_relation)
                        caption_for_loss.append(caption[instance_idx])
                elif len(curr_relation[0]) == 4:
                    for relation_index, each_relation in enumerate(curr_relation):
                        gt_xy_gmm_loss.append(each_relation)
                        generate_xy_gmm_loss.append(coarse_gmm[instance_idx][object_index[instance_idx][relation_index]])

            # Relative objective loss here
            for pp in range(len(coarse_box_output_for_loss)//2):
                real_loss += self.customized_hinge_loss(coarse_gmm_output_for_loss[2*pp], coarse_gmm_output_for_loss[2*pp+1], relation_for_loss[pp], full_relation_for_loss[pp], caption_for_loss[pp])
                if relation_for_loss[pp] == "above":
                    above_loss += self.customized_hinge_loss(coarse_gmm_output_for_loss[2*pp], coarse_gmm_output_for_loss[2*pp+1], relation_for_loss[pp], full_relation_for_loss[pp], caption_for_loss[pp])
                    cnts[0] += 1
                elif relation_for_loss[pp] == "below":
                    below_loss += self.customized_hinge_loss(coarse_gmm_output_for_loss[2*pp], coarse_gmm_output_for_loss[2*pp+1], relation_for_loss[pp], full_relation_for_loss[pp], caption_for_loss[pp])
                    cnts[1] += 1
                elif relation_for_loss[pp] == "left of":
                    left_loss += self.customized_hinge_loss(coarse_gmm_output_for_loss[2*pp], coarse_gmm_output_for_loss[2*pp+1], relation_for_loss[pp], full_relation_for_loss[pp], caption_for_loss[pp])
                    cnts[2] += 1
                elif relation_for_loss[pp] == "right of":
                    right_loss += self.customized_hinge_loss(coarse_gmm_output_for_loss[2*pp], coarse_gmm_output_for_loss[2*pp+1], relation_for_loss[pp], full_relation_for_loss[pp], caption_for_loss[pp])
                    cnts[3] += 1
                else:
                    assert 0
            
            # Absolute objective loss here
            for pp in range(len(generate_xy_gmm_loss)):
                gmm_loss += self.customized_gmm_loss(generate_xy_gmm_loss[pp], gt_xy_gmm_loss[pp])[0]

            num_loss_boxes = len(coarse_box_output_for_loss)
            assert num_loss_boxes != 0

            box_loss = 0
            kl_loss = 0
            box_loss /= num_loss_boxes
            kl_loss /= num_loss_boxes

            rel_loss = 0
            rel2_loss = 0
            
            tot_rel_box_loss = torch.tensor(0.).cuda()
            tot_rel2_box_loss = torch.tensor(0.).cuda()
            tot_coar_box_loss = torch.tensor(0.).cuda()
            tot_coar_kl_loss = torch.tensor(0.).cuda()
            if not self.pretrain_encoder:
                tot_coar_box_loss += box_loss * self.cfg['MODEL']['LOSS']['WEIGHT_COARSE_BOX_LOSS']
                tot_coar_kl_loss += kl_loss * self.cfg['MODEL']['LOSS']['WEIGHT_COARSE_BOX_LOSS'] * 0.1
                tot_rel_box_loss += rel_loss 
                tot_rel2_box_loss += rel2_loss * self.cfg['MODEL']['LOSS']['WEIGHT_COARSE_BOX_LOSS']


            loss = tot_coar_box_loss
            
            if phase == 'train':
                self.encoder_optimizer.zero_grad()
                self.bbox_head_optimizer.zero_grad()
                try:
                    total_loss = real_loss + 0.1*gmm_loss
                    total_loss.backward()
                except:
                    pass
                self.encoder_scheduler.step_and_update_lr()
                if not self.pretrain_encoder:
                    self.bbox_head_scheduler.step_and_update_lr()
            
            if not self.pretrain_encoder:
                coarse_miou += 0
                refine_miou += 0
            
            total_loss += loss.item()
            if not self.pretrain_encoder:
                total_rel_box_loss += tot_rel_box_loss.item()
                total_coar_box_loss += (tot_coar_box_loss.item() + tot_coar_kl_loss.item())
            if phase == 'train':
                if batch_idx % self.cfg['OUTPUT']['NUM_STEPS_SHOW_LOSS'] == 0:
                    self.logger.info('[%d/%d] Loss: %.4f, %.4f'% (batch_idx + 1, len(dataloader), real_loss.item(), gmm_loss.item()))
            elif phase == 'valid':
                print("Validation")
                self.logger.info('[%d/%d] Loss: %.4f, %.4f Loss_box: [%.4f,%.4f]'% (above_loss, below_loss, left_loss, right_loss, batch_idx + 1, len(dataloader), real_loss.item(), gmm_loss.item()))

        log = {
            'epoch': epoch,
            phase + '_loss': total_loss,
            phase + '_coarse_miou': 0.0,
            phase + '_refine_miou': 0.0
        }

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
    
    def _log_epoch(self, epoch, total_loss, total_vocab_loss, total_obj_id_loss,
                   total_token_type_loss, total_rel_box_loss,
                   total_coar_box_loss, total_refi_box_loss, coarse_miou, refine_miou,
                   acc, acc_id, acc_type, phase, encoder_optimizer, bbox_head_optimizer):
        
        log = {
            'epoch': epoch,
            phase + '_loss': total_loss,
            phase + '_vocab_loss': total_vocab_loss,
            phase + '_obj_id_loss': total_obj_id_loss,
            phase + '_token_type_loss': total_token_type_loss,
            phase + '_rel_box_loss': total_rel_box_loss,
            phase + '_coar_box_loss': total_coar_box_loss,
            phase + '_refi_box_loss': total_refi_box_loss,
            phase + '_coarse_miou': coarse_miou,
            phase + '_refine_miou': refine_miou
        }
        self.tb_writer.add_scalar( phase + "/Loss", total_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_vocab", total_vocab_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_obj_id", total_obj_id_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_token_type", total_token_type_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_rel_box", total_rel_box_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_coar_box", total_coar_box_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Loss_refi_box", total_refi_box_loss, epoch)
        self.tb_writer.add_scalar( phase + "/Coarse_miou", coarse_miou, epoch)
        self.tb_writer.add_scalar( phase + "/Refine_miou", refine_miou, epoch)

        self.tb_writer.add_scalar( phase + "/mask_acc", acc, epoch)
        self.tb_writer.add_scalar( phase + "/obj_id_acc", acc_id, epoch)
        self.tb_writer.add_scalar( phase + "/type_acc", acc_type, epoch)
        self.tb_writer.add_scalar( phase + "/enc_lr", encoder_optimizer.param_groups[0]['lr'], epoch)
        self.tb_writer.add_scalar( phase + "/box_lr", bbox_head_optimizer.param_groups[0]['lr'], epoch)
        self.logger.info('[TOTAL] Loss: %.4f'%(total_loss))
        self.logger.info('[TOTAL] Coarse_mIOU: %.4f Refine_mIOU: %.4f'%(coarse_miou, refine_miou))
        self.logger.info("[TOTAL] Mask word acc: %.4f"%(acc))
        self.logger.info("[TOTAL] Mask obj_id acc: %.4f"%(acc_id))
        self.logger.info("[TOTAL] Mask type acc: %.4f"%(acc_type))
        self.logger.debug("="*30)

        return log

    def idx2vocab(self, idx, modality):
        sent = ""
        for i in range(len(idx)):
            if modality == 'text' or modality == 0:
                sent += self.vocab_dict[idx[i]]
            elif modality == 'image' or modality == 1:
                sent += self.cls_dict[idx[i]]
            sent += " "
        return sent

    def build_loss(self):
        rel_gt = self.cfg['MODEL']['ENCODER']['REL_GT']
        raw_batch_size = self.cfg['SOLVER']['BATCH_SIZE']
        KD_ON = self.cfg['MODEL']['LOSS']['KD_LOSS']
        Topk = self.cfg['MODEL']['LOSS']['TOPK']
        if self.cfg['MODEL']['DECODER']['BOX_LOSS'] == 'PDF':
            box_loss = Log_Pdf(reduction='sum',pretrain = True, lambda_xy = 1., lambda_wh = 1., rel_gt = rel_gt, raw_batch_size=raw_batch_size, KD_ON=KD_ON, Topk=Topk)
            rel_loss = Rel_Loss(reduction = 'sum', raw_batch_size=raw_batch_size)
        else:
            box_loss = RegLoss(reduction='sum',pretrain = True, lambda_xy = 1., lambda_wh = 1.)
            rel_loss = None
        gmm_loss_func = Customized_Gmm_Loss()
        box_loss_func = Customized_Box_Loss()
        hinge_loss_func = Customized_Hinge_Loss()

        refine = self.cfg['MODEL']['REFINE']['REFINE']
        if refine:
            if self.cfg['MODEL']['REFINE']['BOX_LOSS'] == 'PDF':
                refine_box_loss = Log_Pdf(reduction='sum',pretrain = True, lambda_xy = 1., lambda_wh = 1., rel_gt = rel_gt, raw_batch_size=raw_batch_size, KD_ON=KD_ON, Topk=Topk)

            else:
                refine_box_loss = RegLoss(reduction='sum',pretrain = True, lambda_xy = 1., lambda_wh = 1., refine=True)


            return refine, box_loss, rel_loss, refine_box_loss, gmm_loss_func, box_loss_func, hinge_loss_func
        else:
            return refine, box_loss, rel_loss, None


    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        torch.backends.cudnn.benchmark = False
        return device

    def _resume_checkpoint(self, path):
        if path == None: return
        try:
            checkpoint = torch.load(path)
            try:
                self.model.load_state_dict(checkpoint['state_dict']).to(self.device)
            except:
                self.logger.info('[Resume] Only load some ckpt from checkpoint')
                pretrain_ckpt = {k: v for k, v in checkpoint['state_dict'].items()
                                 if 'bbox_head' not in k}
                self.model.load_state_dict(pretrain_ckpt, strict=False).to(self.device)

            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            self.bbox_head_optimizer.load_state_dict(checkpoint['bbox_head_optimizer'])
            self.encoder_scheduler.load_state_dict(checkpoint['n_steps'])
            self.bbox_head_scheduler.load_state_dict(checkpoint['n_steps'])
            self.begin_epoch = checkpoint['log'][-1]['epoch'] + 1
            self.all_log = checkpoint['log']
        except:
            self.logger.error('[Resume] Cannot load from checkpoint')

    def _load_encoder_weight(self, path):
        if path == None: return
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        except Exception as e:
            print(e)
            self.logger.error('[Resume] Cannot load from checkpoint')
