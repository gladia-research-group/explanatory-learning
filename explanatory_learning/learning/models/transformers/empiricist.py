import math
import warnings

import torch
from torch import nn
from torch.nn.modules import LayerNorm, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from torch.nn.init import xavier_uniform_
import pytorch_lightning as pl

from .misc import  PositionalEncoding, VisualizerEncoderLayer, VisualizerDecoderLayer
from explanatory_learning.learning.models import get_piece_embeddings, get_table_embeddings, get_rule_embeddings

class EmpiricistModel(nn.Module):
    """Empiricist Model.

    This model takes as input a batch of tables and structures :math:`[(T_1,s_1), \dots,  
    (T_k, s_k)]`, and tags each structure :math:`s_i` with respect to a rule consistent with the table :math:`T_i`.
    
    .. image:: /imgs/empiricist.png

    The architecture is a transformer encoder-decoder, passing as input to the decoder the table (as a sequence of encoded fixed-length structures) and to
    the decoder the structure (as a sequence of pieces). A classification token is appended to the structure
    in order to predict the structure tag.

    :param structure_length: length of the structures in the table
    :param ninp: number of features in input vectors
    :param nhead: number of heads in multi-head-attention
    :param nhid: number of features in the dense layer after multi-head-attention
    :param nlayers: number of transformer layers (both for encoder and decoder)
    :param dropout: dropout of the transformer layers
    :param cache_attention: if true the model will cache the attention matrices, 
        which can later be used for visualization purposes.
    """
    def __init__(self,
                 structure_length,
                 ninp, nhead, nhid, 
                 nlayers=2, 
                 dropout=0.1,
                 cache_attention=False,
                 **kwargs):
        super().__init__()
        
        if len(kwargs) != 0:
            unexp_args = kwargs.keys()
            warnings.warn(f"unexpected input arguments [ignored]: {unexp_args}")
                
        self.structure_embeddings = get_table_embeddings(embedding_dim=ninp, structure_length=structure_length)
        self.piece_embeddings = get_piece_embeddings(embedding_dim=ninp)

        self.pos_encoder = PositionalEncoding(ninp, dropout)
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

    def forward(self, tables, structures, table_labels): 

        #tables is of size [TABLE_SIZE, BATCH_SIZE, STRUCTURE_LENGTH] and type long
        #tables_labels is of size [TABLE_SIZE, BATCH_SIZE] and type float32
        #structure is of size  [STRUCTURE_LENGTH, BATCH_SIZE] and type long

        assert len(tables.shape) == 3 and tables.dtype == torch.long
        assert len(structures.shape) == 2 and structures.dtype == torch.long
        assert len(table_labels.shape) == 2 and table_labels.dtype == torch.long

        table_size, batch_size, structure_length = tables.shape
        structure_length, batch_size_2 =  structures.shape
        assert batch_size == batch_size_2

        device = tables.device

        # encode input structure and table
        labeled_table = self.structure_embeddings(tables, table_labels) #NOTE * math.sqrt(self.ninp)
        
        structures = self.piece_embeddings(structures) 
        structures = self.pos_encoder(structures)#NOTE * math.sqrt(self.ninp)

        token = self.cls_token_embedding.repeat([1,batch_size,1])
        structures = torch.cat([token,structures], dim=0)

        # apply transformers
        trasnformer_output = self.transformer(src=labeled_table, tgt=structures)
        trasnformer_output = trasnformer_output[0,:,:].view(batch_size,self.ninp)
        logits = self.final_fc(trasnformer_output)
        return logits

class LitEmpiricistModel(pl.LightningModule):
    """Pytorch Lightning wrapper for the  :class:`EmpiricistModel`.

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
        self.model  = EmpiricistModel(**kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, tables, structures, table_labels): 
        output = self.model(
            tables=tables, 
            structures=structures,
            table_labels=table_labels)
        return output
    
    # Pytorch Lightning functions ----------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx:int):
        (tables, table_labels, structures, labels) = batch
        structure_length, batch_size = structures.shape

        output = self(
            tables=tables, 
            structures=structures, 
            table_labels=table_labels)
        
        loss = self.criterion(
            input=output.view(batch_size, 2), 
            target=labels.view(batch_size))
        
        # add logging
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx:int):        
        (tables, table_labels, structures, labels) = batch
        structure_length, batch_size  = structures.shape

        output = self(
            tables=tables, 
            structures=structures, 
            table_labels=table_labels)

        loss = self.criterion(
            input=output.view(batch_size, 2), 
            target=labels.view(batch_size))
        
        predicted_labels = output.view(batch_size , 2).argmax(dim=-1).view(batch_size)
        labels = labels.view(batch_size)

        errors = predicted_labels.logical_xor_(labels).sum()
        accuracy = 1 - torch.true_divide(errors, batch_size)

        self.log('val_loss', loss)
        self.log('val_accuracy',accuracy)
        return loss

    def test_step(self, batch, batch_idx:int):
        (tables, table_labels, structures, labels) = batch
        structure_length, batch_size  = structures.shape

        output = self(
            tables=tables, 
            structures=structures, 
            table_labels=table_labels)
                
        predicted_labels = output.view(batch_size , 2).argmax(dim=-1).view(batch_size)
        labels = labels.view(batch_size)

        errors = predicted_labels.logical_xor_(labels).sum()
        accuracy = 1 - torch.true_divide(errors, batch_size)

        self.log("test_accuracy", accuracy)


###############################################################################
#------------------------------------------------------------------------------
###############################################################################

class AwareEmpiricistModel(nn.Module):
    """Empiricist Model.

    This model takes as input a batch of tables,  structures, and rules :math:`[(T_1,s_1,r_1), \dots,  
    (T_k, s_k,r_k)]`, and tags each structure :math:`s_i` with respect to a rule consistent with the table :math:`T_i`, while
    applying autoregression on the input rule :math:`r_i`.
    
    .. image:: /imgs/aware.png

    The architecture is a transformer encoder-decoder, with two decoder layers: one for tag prediction (**structure decoder**)
    and one for rule autoregression (**rule decoder**).
    The input to the encoder is hence the table (as a sequence of encoded fixed-length structures).
    The rule decoder uses the rule itself as both input and output labels (after being shifted right).
    The structure decoder takes a sequence of pieces as input, and a classification token is appended to
    predict the structure tag.

    :param ninp: number of features in input vectors
    :param nhead: number of heads in multi-head-attention
    :param nhid: number of features in the dense layer after multi-head-attention
    :param nlayers: number of transformer layers (both for encoder and decoder)
    :param dropout: dropout of the transformer layers
    :param cache_attention: if true the model will cache the attention matrices, 
        which can later be used for visualization purposes.
    """
    def __init__(self,
                 structure_length,
                 ninp, 
                 nhead, 
                 nhid, 
                 nlayers=2, 
                 dropout=0.1, #table_encoder=TableEncoderRandom,
                 cache_attention=False,
                 **kwargs):
        super().__init__()
        
        if len(kwargs) != 0:
            unexp_args = kwargs.keys()
            warnings.warn(f"unexpected input arguments [ignored]: {unexp_args}")
        
        
        self.structure_embeddings = get_table_embeddings(embedding_dim=ninp, structure_length=structure_length)
        self.piece_embeddings = get_piece_embeddings(embedding_dim=ninp)
        self.rule_embeddings = get_rule_embeddings(embedding_dim=ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.num_tokens=self.rule_embeddings.num_words
        self.final_fc_structures = nn.Linear(ninp, 2)
        self.final_fc_rules = nn.Linear(ninp, self.num_tokens)
        self.ninp = ninp
        self.rules_mask = None

        if cache_attention:
            encoder_layer = VisualizerEncoderLayer(ninp, nhead, nhid, dropout,  "relu")
            decoder_layer_structures = VisualizerDecoderLayer(ninp, nhead, nhid, dropout, "relu")
            decoder_layer_rules = VisualizerDecoderLayer(ninp, nhead, nhid, dropout, "relu")
        else:
            encoder_layer = TransformerEncoderLayer(ninp, nhead, nhid, dropout, "relu")
            decoder_layer_structures = TransformerDecoderLayer(ninp, nhead, nhid, dropout, "relu")
            decoder_layer_rules = TransformerDecoderLayer(ninp, nhead, nhid, dropout, "relu")
        
        encoder_norm = LayerNorm(ninp)
        self.encoder = TransformerEncoder(encoder_layer, nlayers, encoder_norm)

        decoder_norm_structures = LayerNorm(ninp)
        self.decoder_structures = TransformerDecoder(decoder_layer_structures, nlayers, decoder_norm_structures)

        decoder_norm_rules = LayerNorm(ninp)
        self.decoder_rules = TransformerDecoder(decoder_layer_rules, nlayers, decoder_norm_rules)
        
        cls_embedding = torch.normal(mean=0, std=1, size=[1, 1, self.ninp])
        self.cls_token_embedding = torch.nn.parameter.Parameter(cls_embedding)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1: xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, tables, table_labels, structures=None, rules=None):

        #tables is of size [TABLE_SIZE, BATCH_SIZE, STRUCTURE_LENGTH] and type long
        #tables_labels is of size [TABLE_SIZE, BATCH_SIZE] and type long
        #structure is of size  [STRUCTURE_LENGTH, BATCH_SIZE] and type long
        #rule is of size  [RULE_LENGTH, BATCH_SIZE] and type long

        assert len(tables.shape) == 3 and tables.dtype == torch.long
        assert len(table_labels.shape) == 2 and table_labels.dtype == torch.long
        
        table_size, batch_size, structure_length = tables.shape
        device = tables.device
        
        # encode input structure and table
        labeled_table = self.structure_embeddings(tables, table_labels)#NOTE *math.sqrt(self.ninp)
        memory = self.encoder(labeled_table)

        assert structures is not None or rules is not None #<- at least one decoder should work

        if structures is not None:
            assert len(structures.shape) == 2 and structures.dtype == torch.long
            structure_length_structs, batch_size_structs = structures.shape
            assert batch_size == batch_size_structs

            structures = self.piece_embeddings(structures)
            structures = self.pos_encoder(structures)# NOTE * math.sqrt(self.ninp)
            cls_token = self.cls_token_embedding.repeat([1,batch_size,1])
            structures = torch.cat([cls_token, structures], dim=0)

            # apply transformers and final layer
            hidden_structures = self.decoder_structures(structures, memory)[0,:,:].view(batch_size, self.ninp)
            logits_structures = self.final_fc_structures(hidden_structures)
        else:
            logits_structures = None


        if rules is not None:
            assert len(rules.shape) == 2 and table_labels.dtype == torch.long
            rule_length, batch_size_rules = rules.shape
            assert batch_size == batch_size_rules

            # rule mask
            if self.rules_mask is None or self.rules_mask.size(0) < rule_length:
                self.rules_mask = self._generate_square_subsequent_mask(rule_length).to(device)
            mask = self.rules_mask[:rule_length,:rule_length]

            rules = self.rule_embeddings(rules)
            rules = self.pos_encoder(rules)* math.sqrt(self.ninp)

            # apply transformers and final layer
            output_rules = self.decoder_rules(rules, memory, tgt_mask=mask)
            logits_rules = self.final_fc_rules(output_rules)
        else:
            logits_rules = None

        return logits_structures, logits_rules

class LitAwareEmpiricistModel(pl.LightningModule):
    """Pytorch Lightning wrapper for the  :class:`AwareEmpiricistModel`.

    :param learning_rate: learning rate during optimization
    :param rule_reg_coeff: scaling coefficient on rule loss
    :param ninp: number of features in input vectors
    :param nhead: number of heads in multi-head-attention
    :param nhid: number of features in the dense layer after multi-head-attention
    :param nlayers: number of transformer layers (both for encoder and decoder)
    :param dropout: dropout of the transformer layers
    :param cache_attention: if true the model will cache the attention matrices, 
        which can later be used for visualization purposes.
    """
    def __init__(self, learning_rate=1e-4, rule_reg_coeff:float=1.0, **kwargs):
        
        super().__init__()
        self.model  = AwareEmpiricistModel(**kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.rule_reg_coeff = rule_reg_coeff
        self.save_hyperparameters()

    def forward(self, *args,**kwargs):
        logits_structures, logits_rules = self.model(*args,**kwargs)
        return logits_structures, logits_rules
    
    # Pytorch Lightning functions ----------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx:int):
        (tables, table_labels, structures, labels, rules) = batch
        table_size, batch_size, structure_length = tables.shape

        output_structs, output_rules = self(
            tables=tables, 
            structures=structures, 
            table_labels=table_labels,
            rules=rules)
        
        if output_structs is not None:
            loss_structures = self.criterion(
                input=output_structs.view(batch_size, 2), 
                target=labels.view(batch_size))
        else :
            loss_structures = 0

        if output_rules is not None:
            target_rules = rules.roll(shifts=-1,dims=0)
            num_tokens = self.model.num_tokens
            loss_rules = self.criterion(
                input=output_rules.view(-1, num_tokens), 
                target=target_rules.view(-1))
        else:
            loss_rules = 0
            
        loss = loss_structures + self.rule_reg_coeff*loss_rules
        
        # add logging
        self.log('train_loss_structures', loss_structures)
        self.log('train_loss_rules', loss_rules)
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx:int):        
        (tables, table_labels, structures, labels, rules) = batch
        table_size, batch_size, structure_length = tables.shape

        output_structs, output_rules = self(
            tables=tables, 
            structures=structures, 
            table_labels=table_labels,
            rules=rules)
        
        if output_structs is not None:
            loss_structures = self.criterion(
                input=output_structs.view(batch_size, 2), 
                target=labels.view(batch_size))
            predicted_labels = output_structs.view(batch_size , 2).argmax(dim=-1).view(batch_size)
            labels = labels.view(batch_size)

            errors = predicted_labels.logical_xor_(labels).sum()
            structure_accuracy = 1 - torch.true_divide(errors, batch_size)

            self.log('val_loss_structures', loss_structures)
            self.log('val_accuracy_structures', structure_accuracy)
        else:
            loss_structures = 0

        if output_rules is not None:
            target_rules = rules.roll(shifts=-1,dims=0).view(-1)
            num_tokens = self.model.num_tokens
            loss_rules = self.criterion(
                input=output_rules.view(-1, num_tokens),
                target=target_rules)
            
            rule_length, batch_size = output_rules.shape[0], output_rules.shape[1]
            predicted_tokens = output_rules.view(rule_length*batch_size, num_tokens).argmax(dim=-1).view(rule_length*batch_size)
            rule_accuracy = torch.true_divide((predicted_tokens == target_rules).sum(), rule_length*batch_size)

            self.log('val_loss_rules', loss_rules)
            self.log('val_accuracy_rules', rule_accuracy)
        else:
            loss_rules = 0

        loss = loss_structures + self.rule_reg_coeff*loss_rules
        return loss

    def test_step(self, batch, batch_idx:int):
        (tables, table_labels, structures, labels, rules) = batch
        table_size, batch_size, structure_length = tables.shape

        output_structs, output_rules = self(
            tables=tables, 
            structures=structures, 
            table_labels=table_labels,
            rules=rules)
        
        if output_structs is not None:
            loss_structures = self.criterion(
                input=output_structs.view(batch_size, 2), 
                target=labels.view(batch_size))
            predicted_labels = output_structs.view(batch_size , 2).argmax(dim=-1).view(batch_size)
            labels = labels.view(batch_size)

            errors = predicted_labels.logical_xor_(labels).sum()
            structure_accuracy = 1 - torch.true_divide(errors, batch_size)

            self.log('test_loss_structures', loss_structures)
            self.log('test_accuracy_structures', structure_accuracy)
        else:
            loss_structures = 0

        if output_rules is not None:
            target_rules = rules.roll(shifts=-1,dims=0).view(-1)
            num_tokens = self.model.num_tokens
            loss_rules = self.criterion(
                input=output_rules.view(-1, num_tokens),
                target=target_rules)
            
            rule_length, batch_size = output_rules.shape[0], output_rules.shape[1]
            predicted_tokens = output_rules.view(rule_length*batch_size, num_tokens).argmax(dim=-1).view(rule_length*batch_size)
            rule_accuracy = torch.true_divide((predicted_tokens == target_rules).sum(), rule_length*batch_size)

            self.log('test_loss_rules', loss_rules)
            self.log('test_accuracy_rules', rule_accuracy)
        else:
            loss_rules = 0

        loss = loss_structures + self.rule_reg_coeff*loss_rules
        return loss



class LitEmpiricistRuleModel(pl.LightningModule):
    """Pytorch Lightning wrapper for the Empiricist Rule class.

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
        self.model  = AwareEmpiricistModel(**kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, *args,**kwargs):
        logits_structures, logits_rules = self.model(*args,**kwargs)
        return logits_structures, logits_rules
    
    # Pytorch Lightning functions ----------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx:int):
        (tables, table_labels, structures, labels, rules) = batch
        table_size, batch_size, structure_length = tables.shape

        output_structs, output_rules = self(
            tables=tables, 
            structures=None, 
            table_labels=table_labels,
            rules=rules)
        
        assert output_structs is None
        assert output_rules is not None

        target_rules = rules.roll(shifts=-1,dims=0)
        num_tokens = self.model.num_tokens
        loss_rules = self.criterion(
            input=output_rules.view(-1, num_tokens), 
            target=target_rules.view(-1))
            
        loss = loss_rules
        
        # add logging
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx:int):        
        (tables, table_labels, structures, labels, rules) = batch
        table_size, batch_size, structure_length = tables.shape

        output_structs, output_rules = self(
            tables=tables, 
            structures=None, 
            table_labels=table_labels,
            rules=rules)
        
        assert output_structs is None
        assert output_rules is not None

        target_rules = rules.roll(shifts=-1,dims=0).view(-1)
        num_tokens = self.model.num_tokens
        loss_rules = self.criterion(
            input=output_rules.view(-1, num_tokens),
            target=target_rules)
        
        rule_length, batch_size = output_rules.shape[0], output_rules.shape[1]
        predicted_tokens = output_rules.view(rule_length*batch_size, num_tokens).argmax(dim=-1).view(rule_length*batch_size)
        rule_accuracy = torch.true_divide((predicted_tokens == target_rules).sum(), rule_length*batch_size)

        self.log('val_loss', loss_rules)
        self.log('val_accuracy', rule_accuracy)

        loss = loss_rules
        return loss

    def test_step(self, batch, batch_idx:int):
        (tables, table_labels, structures, labels, rules) = batch
        table_size, batch_size, structure_length = tables.shape

        output_structs, output_rules = self(
            tables=tables,
            structures=None,
            table_labels=table_labels,
            rules=rules)

        assert output_structs is None
        assert output_rules is not None

        target_rules = rules.roll(shifts=-1,dims=0).view(-1)
        num_tokens = self.model.num_tokens
        loss_rules = self.criterion(
            input=output_rules.view(-1, num_tokens),
            target=target_rules)
        
        rule_length, batch_size = output_rules.shape[0], output_rules.shape[1]
        predicted_tokens = output_rules.view(rule_length*batch_size, num_tokens).argmax(dim=-1).view(rule_length*batch_size)
        rule_accuracy = torch.true_divide((predicted_tokens == target_rules).sum(), rule_length*batch_size)

        self.log('test_loss', loss_rules)
        self.log('test_accuracy', rule_accuracy)

        loss = loss_rules
        return loss