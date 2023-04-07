import os
import logging
from .base_data_loader import BaseDataLoader
from torch.utils.data.dataloader import default_collate

from .base_data_loader import BaseDataLoader
from .COCODataset import COCORelDataset
from .VGmsdnDataset import VGmsdnRelDataset

class DataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers=0, collate_fn=None):
        if collate_fn is None:
            collate_fn = default_collate
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn)
        
def build_loader(cfg, eval_only):
    logger = logging.getLogger('dataloader')
    
    data_dir = cfg['DATASETS']['DATA_DIR_PATH']
    batch_size = cfg['SOLVER']['BATCH_SIZE']
    batch_size = cfg['SOLVER']['BATCH_SIZE']
    shuffle = cfg['DATALOADER']['SHUFFLE']
    validation_split =cfg['DATALOADER']['VAL_SPLIT']
    num_workers = cfg['DATALOADER']['NUM_WORKER']
    smart_sampling = cfg['DATALOADER']['SMART_SAMPLING']
    is_mask = True
    
    if cfg['DATASETS']['NAME'] == 'coco':
        ins_data_path = os.path.join(data_dir, 'instances_train2017.json')
        sta_data_path = os.path.join(data_dir,'stuff_train2017.json')
        obj_id_v2 = cfg['DATALOADER']['OBJ_ID_MODULE_V2']
        if eval_only:
            ins_data_path = os.path.join(data_dir, 'instances_val2017.json')
            sta_data_path = os.path.join(data_dir,'stuff_val2017.json')
            is_mask = cfg['TEST']['TEST_IS_MASK']
        dataset = COCORelDataset(ins_data_path, sta_data_path, is_mask=is_mask, 
                                 obj_id_v2=obj_id_v2)
        dataset_r = COCORelDataset(ins_data_path, sta_data_path, reverse=True,
                                   is_mask=is_mask)
            
            
    elif cfg['DATASETS']['NAME'] == 'vg_msdn':
        coco_addon = cfg['DATASETS']['COCO_ADDON']
        ins_data_path = os.path.join(data_dir, 'train.json')
        cat_path = os.path.join(data_dir, 'categories.json')
        dict_path = os.path.join(data_dir, 'object_pred_idx_to_name.pkl')
        if eval_only:
            ins_data_path = os.path.join(data_dir, 'test.json')
            is_mask = cfg['TEST']['TEST_IS_MASK']
        dataset = VGmsdnRelDataset(instances_json_path = ins_data_path,
                                       category_json_path = cat_path, 
                                       dict_save_path = dict_path,
                                       add_coco_rel=coco_addon,
                                       sentence_size=128,
                                       is_mask=is_mask)
        dataset_r = VGmsdnRelDataset(instances_json_path = ins_data_path,
                                       category_json_path = cat_path, 
                                       dict_save_path = dict_path,
                                       sentence_size=128, 
                                       add_coco_rel=coco_addon,
                                       reverse=True)

    elif cfg['DATASETS']['NAME'] == 'vg_co':
        ins_data_path = os.path.join(data_dir, 'train.json')
        cat_path = os.path.join(data_dir, 'categories.json')
        dict_path = os.path.join(data_dir, 'object_pred_idx_to_name.pkl')
        if eval_only:
            ins_data_path = os.path.join(data_dir, 'test.json')
        dataset = VGmsdnRelDataset(instances_json_path = ins_data_path,
                                       category_json_path = cat_path, 
                                       dict_save_path = dict_path,
                                       sentence_size=128)          
    else:
        raise Exception("Sorry, we only support coco datasets.")
    
    mode = 'Pretrain' if cfg['MODEL']['PRETRAIN'] else 'Seq2Seq'
    logger.info('Setup [{}] dataset in [{}] mode.'.format(cfg['DATASETS']['NAME'], mode))
    logger.info('[{}] dataset in [{}] mode => Test dataset {}.'.format(cfg['DATASETS']['NAME'], mode, eval_only))
    mycollator=None
    def mycollator(batch):
        return {'bpe_toks': [batch[i][0] for i in range(len(batch))], 'object_index': [batch[i][1] for i in range(len(batch))], 'caption': [batch[i][3] for i in range(len(batch))], \
            'relation': [batch[i][4] for i in range(len(batch))]}
    return DataLoader(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=mycollator), DataLoader(dataset_r, batch_size, shuffle, validation_split, num_workers, collate_fn=mycollator)
