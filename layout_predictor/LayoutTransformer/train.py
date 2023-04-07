from loader import build_loader
from model import build_model
from trainer import build_trainer
from inference import build_inference
from utils import ensure_dir
import logging, coloredlogs
import argparse
import yaml
import os
import torch

# setting parser
parser = argparse.ArgumentParser()
parser.add_argument('--cfg_path', type=str, default='configs/')
parser.add_argument('--default_cfg_path', type=str, default='./configs/default.yaml')
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--eval_only', action='store_true', default=False)
opt = parser.parse_args()

# setting config file
with open(opt.default_cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
with open(opt.cfg_path, 'r') as f:
    cfg.update(yaml.safe_load(f))

# handle dir for saving
ensure_dir(cfg['OUTPUT']['OUTPUT_DIR'])
ensure_dir(cfg['TEST']['OUTPUT_DIR'])

# setting logger
if not opt.eval_only:
    handlers = [logging.FileHandler(os.path.join(cfg['OUTPUT']['OUTPUT_DIR'],'output.log'),
                                    mode = 'w'), logging.StreamHandler()]
else:
    handlers = [logging.FileHandler(os.path.join(cfg['OUTPUT']['OUTPUT_DIR'],
                                   'output_eval.log'), mode = 'w'), logging.StreamHandler()]
logging.basicConfig(handlers = handlers, level=logging.INFO)
logger = logging.getLogger('root')
coloredlogs.install(logger = logger, fmt='%(asctime)s [%(name)s] %(levelname)s %(message)s')
logger.info('Setup output directory - {}.'.format(cfg['OUTPUT']['OUTPUT_DIR']))


if __name__ == '__main__':
    D, D_r = build_loader(cfg, opt.eval_only)
    model = build_model(cfg)
    
    if opt.eval_only:
        assert opt.checkpoint is not None, 'Please provide model ckpt for testing'
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        infer = build_inference(cfg)
        infer.run(cfg=cfg, model=model, dataset=D.dataset)
    else:
        T = build_trainer(cfg=cfg, model=model, dataloader=D, dataloader_r=D_r, opt=opt)
        T.train()