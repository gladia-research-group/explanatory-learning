import math
import warnings
from typing import Dict,  List, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import softmax
import pytorch_lightning as pl

from .misc import  PositionalEncoding, VisualizerEncoderLayer, VisualizerDecoderLayer
from explanatory_learning.learning.models import get_rule_embeddings, get_piece_embeddings
from explanatory_learning.data.encoders import RuleEncoder
from explanatory_learning.learning.samplers import BeamSearcher


class FalsifierModel(nn.Module):
    """Falsifier Model.

    This model takes as input a batch of rules and structures :math:`[(r_1,s_1), \dots,  
    (r_k, s_k)]`, and tags each structure :math:`s_i` with respect to the corresponding rule :math:`r_i`.
    
    .. image:: /imgs/falsifier.png

    The architecture is a transformer encoder-decoder, passing as input to the decoder the rule and to
    the decoder the structure (as a sequence of pieces). A classification token is appended to the structure
    in order to predict the structure tag.

    :param ninp: number of features in input vectors
    :param nhead: number of heads in multi-head-attention
    :param nhid: number of features in the dense layer after multi-head-attention
    :param nlayers: number of transformer layers (both for encoder and decoder)
    :param dropout: dropout of the transformer layers
    :param cache_attention: if true the model will cache the attention matrices, 
        which can later be used for visualization purposes.
    """
    def __init__(self,
                 ninp:int, 
                 nhead:int, 
                 nhid:int,
                 nlayers:int=2,
                 dropout:float=0.1,
                 cache_attention:bool=False,
                 **kwargs):
        super().__init__()
        
        piece_embedding_type = kwargs.pop("piece_embedding_type","bow+bias")

        if len(kwargs) != 0:
            unexp_args = kwargs.keys()
            warnings.warn(f"unexpected input arguments [ignored]: {unexp_args}")
        
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.rule_encoder = get_rule_embeddings(embedding_dim=ninp)
        self.structure_encoder = get_piece_embeddings(embedding_type=piece_embedding_type, embedding_dim=ninp)
        self.final_fc = nn.Linear(ninp, 2)
        self.ninp = ninp

        if cache_attention:
            encoder_layer = VisualizerEncoderLayer(ninp, nhead, nhid, dropout,  "relu")
            encoder_norm = torch.nn.LayerNorm(ninp)
            encoder = torch.nn.TransformerEncoder(encoder_layer, nlayers, encoder_norm)

            decoder_layer = VisualizerDecoderLayer(ninp, nhead, nhid, dropout, "relu")
            decoder_norm = torch.nn.LayerNorm(ninp)
            decoder = torch.nn.TransformerDecoder(decoder_layer, nlayers, decoder_norm)
        else:
            encoder=None
            decoder=None

        self.transformer = nn.Transformer(
            d_model= ninp, 
            nhead = nhead,
            num_encoder_layers = nlayers, 
            num_decoder_layers = nlayers, 
            dim_feedforward = nhid,
            dropout = dropout,
            custom_encoder=encoder,
            custom_decoder=decoder)
        
        cls_embedding = torch.normal(mean=0, std=1, size=[1, 1, self.ninp])
        self.cls_token_embedding = torch.nn.parameter.Parameter(cls_embedding)

        
    def forward(self, rules, structures): 
        """ Forward pass of this network.

        :param rules: tensor containing the input rules. The tensor shape is :math:`(N,B)` with 
            :math:`N` the max rule length in the batch, :math:`B` the batch size
        
        :param structures: tensor containing the input structures. The tensor shape is :math:`(6,B)`
            with :math:`B`the batch size.

        :return: tensor of logits over the tags for the batch. The tensor shape is :math:`(B,2)` with :math:`B` the batch size.
        """
        #rules is of size  [RULE_LENGTH, BATCH_SIZE, TOKEN_EMBEDDING] and type float32
        #structure is of size  [STRUCTURE_LENGTH, BATCH_SIZE, PIECE_EMBEDDING] and type float32

        assert len(rules.shape) == 2 and rules.dtype == torch.long
        assert len(structures.shape) == 2 and structures.dtype == torch.long
        assert structures.shape[1] == rules.shape[1]

        rule_length, batch_size = rules.shape
        structure_length, _ = structures.shape

        # encode rules
        rules = self.rule_encoder(rules) 
        rules = self.pos_encoder(rules) * math.sqrt(self.ninp)

        # encode structures
        structures = self.structure_encoder(structures) 
        structures = self.pos_encoder(structures) * math.sqrt(self.ninp)

        # push classification token
        token = self.cls_token_embedding.repeat([1,batch_size,1])
        structures = torch.cat([token,structures], dim=0)

        # apply transformers
        trasnformer_output = self.transformer(src=rules, tgt=structures)
        trasnformer_output = trasnformer_output[0,:,:].view(batch_size,self.ninp)
        logits = self.final_fc(trasnformer_output)
        return logits

class LitFalsifierModel(pl.LightningModule):
    """Pytorch Lightning variant of :class:`FalsifierModel`.

    :param learning_rate: learning rate during optimization
    :param ninp: number of features in input vectors
    :param nhead: number of heads in multi-head-attention
    :param nhid: number of features in the dense layer after multi-head-attention
    :param nlayers: number of transformer layers (both for encoder and decoder)
    :param dropout: dropout of the transformer layers
    :param cache_attention: if true the model will cache the attention matrices, 
        which can later be used for visualization purposes.
    """
    def __init__(self, learning_rate=1e-4, **kwargs):
        super().__init__()
        self.model  = FalsifierModel(**kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, rules, structures):
        output = self.model(
            rules=rules, structures=structures)
        return output
    
    # Pytorch Lightning functions ----------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx:int):
        (rules, structures, labels) = batch
        structure_length, batch_size = structures.shape
        output = self(rules=rules,structures=structures)

        loss = self.criterion(
            input=output.view(batch_size, 2), 
            target=labels.view(batch_size))
        
        # add logging
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx:int):        
        (rules, structures, labels) = batch
        structure_length, batch_size = structures.shape
        output = self(rules=rules,structures=structures)

        loss = self.criterion(
            input=output.view(batch_size, 2), 
            target=labels.view(batch_size))
        
        predicted_labels = output.view(batch_size , 2).argmax(dim=-1).view(batch_size)
        labels = labels.view(batch_size)

        error_rate = predicted_labels.logical_xor_(labels).to(torch.float32).mean()
        accuracy = 1 - error_rate

        self.log('val_loss', loss)
        self.log('val_accuracy',accuracy)
        return loss

    def test_step(self, batch, batch_idx:int):
        (rules, structures, labels) = batch
        structure_length, batch_size = structures.shape
        output = self(rules=rules,structures=structures)

        predicted_labels = output.view(batch_size , 2).argmax(dim=-1).view(batch_size)
        labels = labels.view(batch_size)
        error_rate = predicted_labels.logical_xor_(labels).to(torch.float32).mean()
        accuracy = 1 - error_rate

        self.log("test_accuracy", accuracy)


class RationalistModel(object):
    def __init__(self, 
                 num_beams:int,
                 falsifier:torch.nn.Module,
                 conjectures_generator:torch.nn.Module,
                 max_rule_length:int,
                 rule_encoder:RuleEncoder):

        super().__init__()
        self.num_beams = num_beams
        self.max_rule_length = max_rule_length

        self.falsifier = falsifier
        self.conjectures_generator = conjectures_generator
        self.rule_encoder = rule_encoder

    def guess_rule(self, table, table_labels, rule:str, find_nearest:bool=False):
        
        #tables is of size [TABLE_SIZE, BATCH_SIZE, STRUCTURE_LENGTH] and type long
        #tables_labels is of size [TABLE_SIZE, BATCH_SIZE] and type long
        
        self.hidden_rule = None
        self.most_probable_rule = None

        beam_searcher = BeamSearcher(
            num_beams=self.num_beams,
            model=self.conjectures_generator,
            model_args_batch_idx={"tables":1, "table_labels":1},
            max_rule_length=self.max_rule_length,
            autoregressive_arg="rules",
            rule_encoder=self.rule_encoder)
        
        rules_encoded, rule_probabilities = beam_searcher.sample(
            tables=table, table_labels=table_labels)

        rules_encoded = rules_encoded.view(self.max_rule_length, -1)
        table_size = table.shape[0]
        self.most_probable_rule = rules_encoded[:, 0:1]

        with torch.no_grad():
            original_rules = rules_encoded

            if find_nearest:
                num_batch_rules = rules_encoded.shape[-1]
                predicted_labels_logits = torch.zeros(
                    [table_size, num_batch_rules, 2], 
                    dtype=torch.float32, device=rules_encoded.device)
                
                for i in range(table_size):
                    structure = table[i,0, :] #get  i-th table structure
                    structure = structure.view(-1, 1).repeat([1, num_batch_rules])
                    structure_label = table_labels[i, 0]

                    falsifier_args = {"rules":rules_encoded, "structures":structure}
                    predicted_labels_logits[i,:,:] = self.falsifier(**falsifier_args).view(num_batch_rules, 2)
                
                predicted_labels = predicted_labels_logits.argmax(dim=-1)
                ham_dist = torch.logical_xor(
                    predicted_labels, table_labels.view(table_size, 1)).sum(dim=0)

                min_value, min_value_index = torch.min(ham_dist, dim=0)
                self.hidden_rule = rules_encoded[:, min_value_index:min_value_index+1]
            else:
                no_rule_found = False
                for i in range(table_size):
                    num_batch_rules = rules_encoded.shape[-1]

                    structure = table[i,0, :] #get  i-th table structure
                    structure = structure.view(-1, 1).repeat([1, num_batch_rules])
                    structure_label = table_labels[i,0]

                    falsifier_args = {"rules":rules_encoded, "structures":structure}
                    
                    predicted_labels_logits = falsifier(**falsifier_args)
                    predicted_labels = predicted_labels_logits.argmax(dim=-1)
                    rules_encoded = rules_encoded[:, predicted_labels == structure_label]
                    if rules_encoded.shape[-1] == 0:
                        no_rule_found = True 
                        break
                
                if not no_rule_found:
                    self.hidden_rule = rules_encoded[:,0:1]

    def predict_labels(self, structures):
        batch_size = structures.shape[1]
        rules = self.hidden_rule.view(-1,1).repeat([1,batch_size])
        structure_labels = self.falsifier(rules=rules, structures=structures).argmax(dim=-1)
        return structure_labels