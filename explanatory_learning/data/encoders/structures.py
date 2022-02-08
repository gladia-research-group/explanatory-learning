from typing import List

PIECE_EMBEDDING_SIZE = 6

# features indices
PYRAMID_IDX = 0
BLOCK_IDX = 1
BLUE_IDX = 2
RED_IDX = 3
POINTUP_IDX = 4
POINTDOWN_IDX = 5


def encode_piece(s: str) -> List[int]:
    if s == ".":
        return [0, 0, 0, 0, 0, 0]
    if s == "a":
        return [1, 0, 1, 0, 1, 0]
    if s == "b":
        return [0, 1, 1, 0, 0, 0]
    if s == "A":
        return [1, 0, 0, 1, 1, 0]
    if s == "B":
        return [0, 1, 0, 1, 0, 0]
    if s == "v":
        return [1, 0, 1, 0, 0, 1]
    if s == "V":
        return [1, 0, 0, 1, 0, 1]
    assert False


def decode_piece(emb: List[int]) -> str:  # <- inverse function
    if emb == [0, 0, 0, 0, 0, 0]:
        return "."
    if emb == [1, 0, 1, 0, 1, 0]:
        return "a"
    if emb == [0, 1, 1, 0, 0, 0]:
        return "b"
    if emb == [1, 0, 0, 1, 1, 0]:
        return "A"
    if emb == [0, 1, 0, 1, 0, 0]:
        return "B"
    if emb == [1, 0, 1, 0, 0, 1]:
        return "v"
    if emb == [1, 0, 0, 1, 0, 1]:
        return "V"
    assert False


def encode_structure(structure: str) -> List[int]:
    out = []
    for s in structure:
        for val in encode_piece(s):
            out.append(val)
    return out


def decode_structure(structure: List[int]) -> str:
    total_length = len(structure)
    assert total_length % PIECE_EMBEDDING_SIZE == 0
    structure_size = total_length // PIECE_EMBEDDING_SIZE
    out = ""
    for i in range(structure_size):
        embedding = structure[i * PIECE_EMBEDDING_SIZE : (i + 1) * PIECE_EMBEDDING_SIZE]
        out += decode_piece(embedding)
    return out


import torch
from typing import List, Optional
from warnings import warn


class StructureEncoder(object):
    """TODO"""

    def __init__(self, pad_token="."):
        super().__init__()
        self.pad = pad_token
        self._tokens = {}
        self._token_idx = []
        self.tokenizer = lambda x: list(x)

    def fit(self, data: List[str]):
        """TODO"""
        tokens = {self.pad}
        for structure in data:
            structure_tokens = self.tokenizer(structure)
            for token in structure_tokens:
                tokens.add(token)

        for i, token in enumerate(sorted(tokens)):
            self._tokens[token] = i
            self._token_idx.append(token)

    def transform(self, structures: List[str]) -> torch.LongTensor:
        """TODO"""
        structures_count = len(structures)
        max_length = max([len(self.tokenizer(structure)) for structure in structures])
        pad_id = self._tokens[self.pad]

        encoded_structures = torch.full(
            [max_length, structures_count], pad_id, dtype=torch.long
        )
        for struct_idx, structure in enumerate(structures):
            structure_tokens = self.tokenizer(structure)
            struct = []
            for piece in structure_tokens:
                struct.append(self._tokens[piece])
            encoded_structures[: len(structure_tokens), struct_idx] = torch.tensor(
                struct, dtype=torch.long
            )

        return encoded_structures

    def inverse_transform(self, encoded_structures: torch.LongTensor) -> List[str]:
        """TODO"""
        max_length, structures_count = encoded_structures.shape
        pad_id = self._tokens[self.pad]
        structures = []
        for ri in range(structures_count):
            structure = []
            for ti in range(max_length):
                token_id = encoded_structures[ti, ri]
                structure.append(self._token_idx[token_id])

            structure_as_str = "".join(structure)
            structures.append(structure_as_str)
        return structures

    def piece_index(self, token: str):
        assert token in self._tokens
        return self._tokens[token]

    @property
    def pieces(self):
        return self._token_idx

    @property
    def num_pieces(self):
        return len(self._tokens)


_structure_encoder: Optional[StructureEncoder] = None


def get_structure_encoder(encoder_type: str = "default") -> StructureEncoder:
    """
    Returns the desired structure encoder, already fitted.

    :param encoder_type: type of encoder wanted.
    :returns: encoder of chosen type already fit with correct data.
    """
    if encoder_type == "default":
        structure_encoder = StructureEncoder()
        structure_encoder.fit(["aA....", "vV....", "bB...."])
        return structure_encoder

    elif encoder_type == "global":
        global _structure_encoder
        if _structure_encoder is None:
            _structure_encoder = get_structure_encoder("default")
        return _structure_encoder
