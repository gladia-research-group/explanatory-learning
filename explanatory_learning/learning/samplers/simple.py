
from typing import Dict,  List, Tuple

import torch
from torch.nn.functional import softmax
from explanatory_learning.data.encoders import RuleEncoder

class Sampler(object):
    def __init__(self, 
                 num_samples:int,
                 model:torch.nn.Module,
                 model_args_batch_idx:Dict[str,int],
                 max_rule_length:int,
                 autoregressive_arg:str,
                 rule_encoder:RuleEncoder):

        super().__init__()
        self.num_samples = num_samples
        self.max_rule_length = max_rule_length
        self.autoregressive_arg = autoregressive_arg

        self.model = model
        self.model_args_batch_idx = model_args_batch_idx
        self.rule_encoder = rule_encoder
        #self.pad_token_idx = self.rule_encoder.token_index(self.rule_encoder.pad)


    @property
    def device(self):
        return self.model.device

    def _next_words(self, 
        model_args:dict,
        next_token_idx:torch.LongTensor,
        rules_encoded:torch.LongTensor,
        rule_probabilities:torch.Tensor):
    
        model_args[self.autoregressive_arg] = rules_encoded
        num_rules = rules_encoded.shape[1]
        arange_num_rules = torch.arange(num_rules, device=self.device)

        logits_structures, logits_rules = self.model(**model_args)
        next_token_logits = logits_rules[next_token_idx, arange_num_rules, :]
        probability_next_tokens = softmax(
            next_token_logits, dim=-1) #<- conditional probabilities for the next token in the sequence
                                       #   shape [num. rules, num. tokens]

        sampled_tokens = torch.multinomial(probability_next_tokens, num_samples=1).view(num_rules) #sample from distribution
        rule_probabilities.mul_(probability_next_tokens[arange_num_rules, sampled_tokens])
        rules_encoded[next_token_idx+1, arange_num_rules] = sampled_tokens

    def sample(self, **model_args):
        batch_size = -1
        
        for k,v in model_args.items():
            tmp_batch_size = v.shape[self.model_args_batch_idx[k]]
            assert batch_size == -1 or batch_size == tmp_batch_size
            batch_size = tmp_batch_size

        tmp = model_args
        model_args = {}
        for k,v in tmp.items():
            model_args[k] = v.to(self.device).repeat_interleave(
                repeats=self.num_samples,
                dim=self.model_args_batch_idx[k])

        next_token_idx = torch.zeros([self.num_samples*batch_size], device=self.device, dtype=torch.long)

        rules_encoded = torch.tensor(
            self.rule_encoder.transform([""]*(self.num_samples*batch_size), min_padding_length=self.max_rule_length - 2), # NOTE do not count for sos and eos
            device=self.device, dtype=torch.long)

        rule_probabilities =  torch.ones([batch_size*self.num_samples], device=self.device)

        with torch.no_grad():
            for i in range(self.max_rule_length - 1):
                
                self._next_words(
                    model_args=model_args,
                    next_token_idx=next_token_idx,
                    rules_encoded=rules_encoded,
                    rule_probabilities=rule_probabilities)
                next_token_idx += 1

            rule_probabilities = rule_probabilities.view(batch_size, self.num_samples)
            rules_encoded = rules_encoded.view(self.max_rule_length, batch_size, self.num_samples)
        return rules_encoded, rule_probabilities
