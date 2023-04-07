import numpy as np
import logging

class Scheduler():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        print(optimizer)
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def resume_checkpoint(self, ckpt_steps):
        self.n_current_steps = ckpt_steps

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            
class ChrisScheduler():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, n_hold_steps, 
                 n_decay_steps, max_lr, min_lr):
        print(optimizer)
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_hold_steps = n_hold_steps
        self.n_decay_steps = n_decay_steps
        self.n_current_steps = 0
        self.max_lr = max_lr
        self.min_lr = min_lr

    def resume_checkpoint(self, ckpt_steps):
        self.n_current_steps = ckpt_steps

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        
        self.n_current_steps += 1
        
        if self.n_current_steps < self.n_warmup_steps: # linear
            lr = 0.1 * self.max_lr + \
            (self.max_lr - 0.1 * self.max_lr) / self.n_warmup_steps * self.n_current_steps
        elif self.n_current_steps >= self.n_warmup_steps and self.n_current_steps < self.n_hold_steps + self.n_warmup_steps: # hold
            lr = self.max_lr
        else: # 2 degree
            A = (self.max_lr * 0.9/1.0)**2. / self.n_decay_steps
            lr = -((self.n_current_steps - (self.n_warmup_steps+self.n_hold_steps))*A)**0.5 + self.max_lr
            
        if lr < self.min_lr:
            lr = self.min_lr
            
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            
class BertScheduler():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, n_hold_steps, 
                 n_decay_steps, max_lr, min_lr):
        print(optimizer)
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_hold_steps = n_hold_steps
        self.n_decay_steps = n_decay_steps
        self.n_current_steps = 0
        self.max_lr = max_lr
        self.min_lr = min_lr

    def resume_checkpoint(self, ckpt_steps):
        self.n_current_steps = ckpt_steps

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        
        self.n_current_steps += 1
        
        if self.n_current_steps < self.n_warmup_steps: # linear
            lr = 0.1 * self.max_lr + \
            (self.max_lr - 0.1 * self.max_lr) / self.n_warmup_steps * self.n_current_steps
        elif self.n_current_steps >= self.n_warmup_steps and self.n_current_steps < self.n_hold_steps + self.n_warmup_steps: # hold
            lr = self.max_lr
        else: # 2 degree
            A = self.max_lr / self.n_decay_steps
            lr = -((self.n_current_steps - (self.n_warmup_steps+self.n_hold_steps))*A) + self.max_lr
            
        if lr < self.min_lr:
            lr = self.min_lr

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

def build_scheduler(cfg, optimizer, total_steps, type):
    logger = logging.getLogger('scheduler')
    if type =='ENC':
        if cfg['SOLVER']['ENCODER']['LR_SCHEDULER'] == 'BaseScheduler':
            S = Scheduler(optimizer, cfg['MODEL']['ENCODER']['HIDDEN_SIZE'], 
                          cfg['SOLVER']['ENCODER']['WARMUP_STEPS'])
        elif cfg['SOLVER']['ENCODER']['LR_SCHEDULER'] == 'ChrisScheduler':
            S = ChrisScheduler(optimizer, cfg['MODEL']['ENCODER']['HIDDEN_SIZE'], 
                         n_warmup_steps=cfg['SOLVER']['ENCODER']['WARMUP_RATIO'] * total_steps, 
                         n_hold_steps=cfg['SOLVER']['ENCODER']['HOLD_RATIO'] * total_steps, 
                         n_decay_steps=cfg['SOLVER']['ENCODER']['DECAY_RATIO'] * total_steps, 
                         max_lr=cfg['SOLVER']['ENCODER']['MAX_LR'],
                         min_lr=cfg['SOLVER']['ENCODER']['MIN_LR'])
        elif cfg['SOLVER']['ENCODER']['LR_SCHEDULER'] == 'BertScheduler':
            S = BertScheduler(optimizer, cfg['MODEL']['ENCODER']['HIDDEN_SIZE'], 
                             n_warmup_steps=cfg['SOLVER']['ENCODER']['WARMUP_RATIO'] * total_steps, 
                             n_hold_steps=cfg['SOLVER']['ENCODER']['HOLD_RATIO'] * total_steps, 
                             n_decay_steps=cfg['SOLVER']['ENCODER']['DECAY_RATIO'] * total_steps, 
                             max_lr=cfg['SOLVER']['ENCODER']['MAX_LR'],
                             min_lr=cfg['SOLVER']['ENCODER']['MIN_LR'])
    elif type =='BOX':
        if cfg['SOLVER']['BBOX_HEAD']['LR_SCHEDULER'] == 'BaseScheduler':
            S = Scheduler(optimizer, cfg['MODEL']['ENCODER']['HIDDEN_SIZE'], 
                          cfg['SOLVER']['BBOX_HEAD']['WARMUP_STEPS'])
        elif cfg['SOLVER']['BBOX_HEAD']['LR_SCHEDULER'] == 'ChrisScheduler':
            S = ChrisScheduler(optimizer, cfg['MODEL']['ENCODER']['HIDDEN_SIZE'], 
                         n_warmup_steps=cfg['SOLVER']['BBOX_HEAD']['WARMUP_RATIO'] * total_steps, 
                         n_hold_steps=cfg['SOLVER']['BBOX_HEAD']['HOLD_RATIO'] * total_steps, 
                         n_decay_steps=cfg['SOLVER']['BBOX_HEAD']['DECAY_RATIO'] * total_steps, 
                         max_lr=cfg['SOLVER']['BBOX_HEAD']['MAX_LR'],
                         min_lr=cfg['SOLVER']['BBOX_HEAD']['MIN_LR'])
        elif cfg['SOLVER']['BBOX_HEAD']['LR_SCHEDULER'] == 'BertScheduler':
            S = BertScheduler(optimizer, cfg['MODEL']['ENCODER']['HIDDEN_SIZE'], 
                             n_warmup_steps=cfg['SOLVER']['BBOX_HEAD']['WARMUP_RATIO'] * total_steps, 
                             n_hold_steps=cfg['SOLVER']['BBOX_HEAD']['HOLD_RATIO'] * total_steps, 
                             n_decay_steps=cfg['SOLVER']['BBOX_HEAD']['DECAY_RATIO'] * total_steps, 
                             max_lr=cfg['SOLVER']['BBOX_HEAD']['MAX_LR'],
                             min_lr=cfg['SOLVER']['BBOX_HEAD']['MIN_LR'])        
    else:
        raise Exception("Sorry, we only support BaseScheduler, ChrisScheduler and BertScheduler. Please build your own scheduler.")
    logger.info('Setup scheduler {}.'.format(S.__class__.__name__))
    return S
