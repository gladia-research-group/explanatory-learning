from typing import List
from warnings import warn

import torch
from explanatory_learning.data.encoders import get_rule_encoder


class RuleWordsEmbedding(torch.nn.Embedding):
    """Word embedding for rules."""

    def __init__(self, words: List[str], embedding_dim: int = 64, **kwargs):
        for k in kwargs:
            warn(f"Ignoring unexpected input parameter {k}")

        self.num_words = len(words)
        self.embedding_dim = embedding_dim
        super().__init__(
            num_embeddings=self.num_words, embedding_dim=self.embedding_dim
        )

        # this attributes are necessary in order to enforce consistency between embeddings orders
        self._verification_code = encode_words(words)
        self.register_buffer(
            "_verification_code_loaded", self._verification_code
        )  # <- NOTE Obsolete

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        :param x: batch of rules (as a sequence of word indices) with shape :math:`(N,B)`,
            with :math:`N` the length of the longest rule in the batch, and :math:`B` the batch size.
        :return: batch of rules (as sequence of word embeddings) with shape :math:`(N,B,F)` with :math:`F`
            equals to the word embedding size.
        """
        return super().forward(x)


def get_rule_embeddings(
    embedding_dim: int, embedding_type: str = "trainable", **kwargs
) -> RuleWordsEmbedding:
    for k in kwargs.keys():
        warn(f"input parameter {k} was ignored")

    rule_encoder = get_rule_encoder("global")
    words = rule_encoder.tokens

    if embedding_type == "trainable":
        return RuleWordsEmbedding(words=words, embedding_dim=embedding_dim)

    assert False


def encode_words(words: List[str]) -> torch.LongTensor:
    words_joined = "$".join(words)
    iwords = [ord(char) for char in words_joined]
    return torch.tensor(iwords, dtype=torch.long)


def decode_words(words_encoded: torch.LongTensor) -> List[str]:
    words_encoded = words_encoded.clone().detach().cpu().tolist()
    words_joined = "".join([chr(i) for i in words_encoded])
    return words_joined.split("$")
