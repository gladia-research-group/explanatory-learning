from typing import List, Optional

import torch


class RuleEncoder(object):
    def __init__(
        self, init_token="<sos>", eos_token="<eos>", pad_token="<pad>", tokenizer=None
    ):  # TODO add lower
        super().__init__()
        self.sos = init_token
        self.eos = eos_token
        self.pad = pad_token
        self._tokens = {}
        self._token_idx = []
        self.tokenizer = lambda x: x.split() if tokenizer is None else tokenizer

    def fit(self, data: List[str]):
        tokens = {self.sos, self.eos, self.pad}
        for rule in data:
            rule_tokens = self.tokenizer(rule)
            for token in rule_tokens:
                tokens.add(token)

        for i, token in enumerate(sorted(tokens)):
            self._tokens[token] = i
            self._token_idx.append(token)

    def transform(
        self, rules: List[str], min_padding_length: int = 0
    ) -> torch.LongTensor:
        rule_count = len(rules)
        max_length = max([len(self.tokenizer(rule)) for rule in rules])
        max_length = max(max_length, min_padding_length)

        pad_id = self._tokens[self.pad]
        sos_token = [self.sos]  # NOTE set to [] to remove it
        eos_token = [self.eos]  # NOTE set to [] to remove it
        encoded_rules = torch.full(
            [max_length + len(sos_token) + len(eos_token), rule_count],
            pad_id,
            dtype=torch.long,
        )

        for ri, rule in enumerate(rules):
            rule_tokens = sos_token + self.tokenizer(rule) + eos_token
            for ti, token in enumerate(rule_tokens):
                encoded_rules[ti, ri] = self._tokens[token]

        return encoded_rules

    def inverse_transform(
        self, encoded_rules: torch.LongTensor, ignore_tokens: List[str] = None
    ) -> List[str]:
        max_length, rule_count = encoded_rules.shape

        if ignore_tokens is None:
            ignore_tokens = [self.sos, self.eos, self.pad]

        # NOTE: do not make a set since hashing of pytorch not good
        ignore_ids = [self._tokens[t] for t in ignore_tokens]

        rules = []
        for ri in range(rule_count):
            rule = []
            for ti in range(max_length):
                token_id = encoded_rules[ti, ri]
                if token_id not in ignore_ids:
                    rule.append(self._token_idx[token_id])

            rule_str = " ".join(rule)
            rules.append(rule_str)
        return rules

    def token_index(self, token: str):
        assert token in self._tokens
        return self._tokens[token]

    @property
    def num_tokens(self):
        return len(self._tokens)

    @property
    def tokens(self):
        return self._token_idx


_rule_encoder: Optional[RuleEncoder] = None


def get_rule_encoder(encoder_type: str = "default") -> RuleEncoder:
    if encoder_type == "default":
        from explanatory_learning.data.utils import generate_rules

        rule_encoder = RuleEncoder()
        rule_encoder.fit(generate_rules())
        return rule_encoder

    elif encoder_type == "global":
        global _rule_encoder
        if _rule_encoder is None:
            _rule_encoder = get_rule_encoder("default")
        return _rule_encoder
