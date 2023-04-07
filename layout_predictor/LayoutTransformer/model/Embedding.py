import torch
import torch.nn as nn
import torch.nn.functional as F

class Add_Embeddings(nn.Module):
    """Construct the embeddings from cls, position and shape embeddings.
    """

    def __init__(self, 
                 vocab_size=154, 
                 max_position_embeddings=68, 
                 max_shape_type=68,
                 hidden_size=512, 
                 hidden_dropout_prob=0.1):

        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
#         self.id_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.shape_embeddings = nn.Embedding(max_shape_type, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, cat_ids, position_ids, shape_ids):
        
        inputs_embeds = self.word_embeddings(cat_ids)
#         ids_embeds = self.id_embeddings(id_ids)
        position_embeddings = self.position_embeddings(position_ids)
        shape_embeddings = self.shape_embeddings(shape_ids)

        embeddings = inputs_embeds + position_embeddings + shape_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Text_Embedding(nn.Module):
    def __init__(self, 
                 vocab_size=204,
                 obj_classes_size=154,
                 hidden_size=512, 
                 max_rel_pair=17, 
                 max_token_type=4,
                 hidden_dropout_prob=0.1):

        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.obj_id_embeddings = nn.Embedding(obj_classes_size, hidden_size, padding_idx=0)
        self.sentence_type = nn.Embedding(max_rel_pair, hidden_size, padding_idx=0)
        self.token_type = nn.Embedding(max_token_type, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(hidden_dropout_prob)

class Sentence_Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, 
                 vocab_size=204,
                 obj_classes_size=154,
                 hidden_size=512, 
                 max_rel_pair=17, 
                 max_token_type=4,
                 hidden_dropout_prob=0.1):

        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.obj_id_embeddings = nn.Embedding(obj_classes_size, hidden_size, padding_idx=0)
        self.sentence_type = nn.Embedding(max_rel_pair, hidden_size, padding_idx=0)
        self.token_type = nn.Embedding(max_token_type, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_token, input_obj_id, segment_label, token_type):

        inputs_embeds = self.word_embeddings(input_token)
        input_id_embdes = self.obj_id_embeddings(input_obj_id)
        segment_embeddings = self.sentence_type(segment_label)
        token_type_embeddings = self.token_type(token_type)

        embeddings = inputs_embeds + segment_embeddings + input_id_embdes + token_type_embeddings
#         embeddings = inputs_embeds + segment_embeddings + token_type_embeddings
#         embeddings = inputs_embeds + segment_embeddings + input_id_embdes
#         embeddings = inputs_embeds + segment_embeddings 
        embeddings = self.dropout(embeddings)
        return embeddings, inputs_embeds

class Concat_Embeddings(nn.Module):
    """Construct the embeddings from cls, position and shape embeddings.
    """

    def __init__(self, 
                 vocab_size=154, 
                 max_position_embeddings=68, 
                 max_shape_type=68,
                 hidden_size=512,
                 hidden_dropout_prob=0.1):

        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size // 2, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size // 4)
        self.shape_embeddings = nn.Embedding(max_shape_type, hidden_size // 4)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, cat_ids, position_ids, shape_ids):
        
        inputs_embeds = self.word_embeddings(cat_ids)
        position_embeddings = self.position_embeddings(position_ids)
        shape_embeddings = self.shape_embeddings(shape_ids)

        embeddings = torch.cat([inputs_embeds, position_embeddings, shape_embeddings], dim=2)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class ConcatBox_Embeddings(nn.Module):
    """Construct the embeddings from cls, box embeddings.
    """
    def __init__(self, 
                 vocab_size=154, 
                 box_size=4, 
                 hidden_size=512,
                 hidden_dropout_prob=0.1):

        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, (hidden_size//2) - 4, padding_idx=0)
        self.templ_embeddings = nn.Embedding(vocab_size, hidden_size//2 , padding_idx=0)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, cat_ids, box, template):
        inputs_embeds = self.word_embeddings(cat_ids)
        templ_embeds = self.templ_embeddings(template)
        embeddings = torch.cat([inputs_embeds, templ_embeds, box], dim=2)
        embeddings = self.dropout(embeddings)
        return embeddings