from typing import Dict

import torch

from torch.nn.functional import softmax
from explanatory_learning.data.encoders import RuleEncoder


class BeamSearcher(object):
    def __init__(
        self,
        num_beams: int,
        model: torch.nn.Module,
        model_args_batch_idx: Dict[str, int],
        max_rule_length: int,
        autoregressive_arg: str,
        rule_encoder: RuleEncoder,
    ):
        super().__init__()
        self.num_samples = num_beams
        self.max_rule_length = max_rule_length
        self.num_tokens = rule_encoder.num_tokens
        self.autoregressive_arg = autoregressive_arg

        self.model = model
        self.model_args_batch_idx = model_args_batch_idx
        self.rule_encoder = rule_encoder
        self.pad_token_idx = self.rule_encoder.token_index(self.rule_encoder.pad)

    @property
    def device(self):
        return self.model.device

    def _next_words(
        self,
        num_rules: int,
        model_args: dict,
        next_token_idx: torch.LongTensor,
        rules_encoded: torch.LongTensor,
        rule_probabilities: torch.Tensor,
    ) -> int:

        num_tokens = self.rule_encoder.num_tokens

        tmp = model_args
        model_args = {}
        for k, v in tmp.items():
            bi = self.model_args_batch_idx[k]
            model_args[k] = v.narrow(dim=bi, start=0, length=num_rules)

        model_args[self.autoregressive_arg] = rules_encoded.narrow(
            dim=1, start=0, length=num_rules
        )

        logits_structures, logits_rules = self.model(**model_args)
        next_token_logits = logits_rules[
            next_token_idx, torch.arange(num_rules, device=self.device), :
        ]  # <- returns shape [num. rules, num. tokens] (I think)

        # conditional probabilities for the next token in the sequence
        probability_next_tokens = softmax(next_token_logits, dim=-1)  # shape [B,]


        candidates_probability = probability_next_tokens
        candidates_probability.mul_(
            rule_probabilities.narrow(dim=0, start=0, length=num_rules).view(
                num_rules, 1
            )
        )
        new_rule_probabilities, I = candidates_probability.view(
            num_rules * num_tokens
        ).sort(
            descending=True
        )  # <- ranges from [0,num.rules*num.tokens)
        I = I.narrow(
            dim=0, start=0, length=min(num_rules * num_tokens, self.num_samples)
        )

        num_new_rules = I.shape[0]
        new_rule_probabilities = new_rule_probabilities[:num_new_rules]

        token_index = I.remainder(num_tokens)
        rule_index = I.floor_divide_(num_tokens)

        new_rules_encoded = rules_encoded.index_select(index=rule_index, dim=-1)
        new_rules_encoded[
            next_token_idx + 1, torch.arange(num_new_rules, device=self.device)
        ] = token_index
        return num_new_rules, new_rules_encoded, new_rule_probabilities

    def sample(self, **model_args):
        batch_size = 1
        num_tokens = self.rule_encoder.num_tokens

        for k, v in model_args.items():
            tmp_batch_size = v.shape[self.model_args_batch_idx[k]]
            assert batch_size == -1 or batch_size == tmp_batch_size
            batch_size = tmp_batch_size

        tmp = model_args
        model_args = {}
        for k, v in tmp.items():
            model_args[k] = v.to(self.device).repeat_interleave(
                repeats=self.num_samples, dim=self.model_args_batch_idx[k]
            )

        next_token_idx = torch.zeros(
            [self.num_samples * batch_size], device=self.device, dtype=torch.long
        )

        rules_encoded = torch.tensor(
            self.rule_encoder.transform(
                [""] * (self.num_samples * batch_size),
                min_padding_length=self.max_rule_length - 2,
            ),
            device=self.device,
            dtype=torch.long,
        )

        rule_probabilities = torch.zeros([self.num_samples], device=self.device)
        rule_probabilities[0] = 1.0

        num_rules = 1
        with torch.no_grad():
            for i in range(self.max_rule_length - 1):

                next_token_idx = torch.full(
                    [], fill_value=i, device=self.device, dtype=torch.long
                )

                (num_rules, rules_encoded, rule_probabilities) = self._next_words(
                    num_rules=num_rules,
                    model_args=model_args,
                    next_token_idx=next_token_idx,
                    rules_encoded=rules_encoded,
                    rule_probabilities=rule_probabilities,
                )

        rule_probabilities = rule_probabilities.view(batch_size, self.num_samples)
        rules_encoded = rules_encoded.view(
            self.max_rule_length, batch_size, self.num_samples
        )
        return rules_encoded, rule_probabilities
