from typing import List, Optional
from warnings import warn

import torch
import torch.nn.init as init
from explanatory_learning.data.encoders import (BLOCK_IDX, BLUE_IDX,
                                                PIECE_EMBEDDING_SIZE,
                                                POINTDOWN_IDX, POINTUP_IDX,
                                                PYRAMID_IDX, RED_IDX,
                                                get_structure_encoder)


class PieceEmbedding(torch.nn.Embedding):
    """Embedding for pieces of structures."""

    def __init__(self, pieces: List[str], embedding_dim: int = 64, **kwargs):
        for k in kwargs:
            warn(f"Ignoring unexpected input parameter {k}")

        self.num_pieces = len(pieces)
        self.embedding_dim = embedding_dim
        super().__init__(
            num_embeddings=self.num_pieces, embedding_dim=self.embedding_dim
        )

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        :param x: batch of structures (as a sequence of piece indices) with shape :math:`(N,B)`,
            with :math:`N=6`, and :math:`B` the batch size.
        :return: batch of structures (as sequence of piece embeddings) with shape :math:`(N,B,F)` with :math:`F`
            equals to the piece embedding size.
        """
        return super().forward(x)


class PieceEmbeddingBoW(torch.nn.Module):
    def __init__(
        self, pieces: List[str], embedding_dim: int = 64, bias: bool = True, **kwargs
    ):
        super().__init__()

        self.pieces = pieces
        self.num_pieces = len(pieces)
        self.embedding_dim = embedding_dim
        self.use_bias = bias
        self._i2p = {i: p for i, p in enumerate(pieces)}
        self._p2f = {
            ".": [],
            "a": [PYRAMID_IDX, BLUE_IDX, POINTUP_IDX],
            "A": [PYRAMID_IDX, RED_IDX, POINTUP_IDX],
            "v": [PYRAMID_IDX, BLUE_IDX, POINTDOWN_IDX],
            "V": [PYRAMID_IDX, RED_IDX, POINTDOWN_IDX],
            "b": [BLOCK_IDX, BLUE_IDX],
            "B": [BLOCK_IDX, RED_IDX],
        }

        i2f = torch.zeros([self.num_pieces, PIECE_EMBEDDING_SIZE], dtype=torch.float32)
        for i in range(self.num_pieces):
            piece = self._i2p[i]
            BoF_indices = self._p2f[piece]
            for j in BoF_indices:
                i2f[i, j] = 1

        self.register_buffer("_i2f", i2f)
        self.weight = torch.nn.parameter.Parameter(
            torch.ones([embedding_dim, PIECE_EMBEDDING_SIZE]), requires_grad=True
        )
        self.bias = torch.nn.parameter.Parameter(
            torch.ones([embedding_dim]), requires_grad=True
        )
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight)
        init.normal_(self.bias)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        :param x: batch of structures (as a sequence of piece indices) with shape :math:`(N,B)`,
            with :math:`N` the structure length, and :math:`B` the batch size.
        :return: batch of structures (as sequence of piece embeddings) with shape :math:`(N,B,F)` with :math:`F`
            equals to the piece embedding size.
        """
        N, B = x.shape
        tmp = torch.nn.functional.embedding(x, self._i2f).view(N * B, -1)
        bow_embeddings = tmp.matmul(self.weight.t()).view(N, B, -1)
        if self.use_bias:
            bow_embeddings = bow_embeddings + self.bias
        return bow_embeddings


def get_piece_embeddings(
    embedding_dim: int, embedding_type: str = "bow+bias", **kwargs
):
    struct_encoder = get_structure_encoder("global")
    pieces = struct_encoder.pieces

    if embedding_type == "onehot":
        return PieceEmbedding(pieces=pieces, embedding_dim=embedding_dim)
    elif embedding_type == "bow":
        return PieceEmbeddingBoW(pieces=pieces, embedding_dim=embedding_dim, bias=False)
    elif embedding_type == "bow+bias":
        return PieceEmbeddingBoW(pieces=pieces, embedding_dim=embedding_dim, bias=True)

    # <- insert
    assert False


class StructureEmbedding(torch.nn.Embedding):
    """Embeddings for structures."""

    def __init__(
        self,
        pieces: List[str],
        embedding_dim: int = 64,
        structure_length=6,
        max_table_size: int = 300,
        **kwargs,
    ):
        for k in kwargs:
            warn(f"Ignoring unexpected input parameter {k}")

        self.num_pieces = len(pieces)
        self.embedding_dim = embedding_dim
        self.structure_length = structure_length
        self.max_table_size = max_table_size

        super().__init__(
            num_embeddings=self.num_pieces * self.structure_length,
            embedding_dim=self.embedding_dim,
        )

        offset_vec = torch.arange(start=0, end=self.structure_length, dtype=torch.long)
        offset_vec = offset_vec.view(1, 1, self.structure_length) * self.num_pieces
        self.register_buffer("offset_vec", offset_vec)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        :param x: batch of tables (as a sequence of structure indices) with shape :math:`(N,T,B)`,
            with :math:`N` the structure length, :math:`T` the maximum table size in the batch,
            and :math:`B` the batch size.
        :return: batch of structures (as sequence of structure embeddings) with shape :math:`(T,B,F)` with :math:`F`
            equals to the structure embedding size.
        """
        T, B, N = x.shape
        offset_x = x.view(T, B, self.structure_length) + self.offset_vec
        embeddings_pos = super().forward(offset_x)
        return embeddings_pos.sum(dim=2)


class TableEmbedding(StructureEmbedding):
    """Embeddings for tables."""

    def __init__(
        self,
        pieces: List[str],
        embedding_dim: int = 64,
        structure_length=6,
        max_table_size: int = 300,
        **kwargs,
    ):
        super().__init__(
            pieces=pieces,
            embedding_dim=embedding_dim,
            structure_length=structure_length,
            max_table_size=max_table_size,
            **kwargs,
        )
        self.tags_embeddings = torch.nn.Embedding(
            num_embeddings=2, embedding_dim=self.embedding_dim
        )

    def forward(
        self, structures: torch.LongTensor, structure_tags: torch.LongTensor
    ) -> torch.Tensor:
        """
        :param structures: batch of tables (as a sequence of structure indices) with shape :math:`(T,B,N)`,
            with :math:`N` the structure length, :math:`T` the maximum table size in the batch,
            and :math:`B` the batch size.
        :param structure_tags: batch of structure labels, indicating the label of the corresponding tag in ``structures``.
            The sahpe of this parameter is :math:`(T,B)` with :math:`N` the structure length and :math:`T` the maximum table size in the batch.
        :return: batch of tagged structures (as sequence of tagged structure embeddings) with shape :math:`(T,B,F)` with :math:`F`
            equals to the structure embedding size.
        """
        T, B, N = structures.shape
        structure_embeddings = super().forward(structures)
        tagged_structure_embeddings = structure_embeddings + self.tags_embeddings(
            structure_tags.view(T, B)
        )
        return tagged_structure_embeddings


def get_structure_embeddings(
    embedding_dim: int, structure_length: int, embedding_type: str = "onehot", **kwargs
):
    struct_encoder = get_structure_encoder("global")
    pieces = struct_encoder.pieces

    if embedding_type == "onehot":
        return StructureEmbedding(
            pieces=pieces,
            embedding_dim=embedding_dim,
            structure_length=structure_length,
        )


def get_table_embeddings(
    embedding_dim: int, structure_length: int, embedding_type: str = "onehot", **kwargs
):
    struct_encoder = get_structure_encoder("global")
    pieces = struct_encoder.pieces

    if embedding_type == "onehot":
        return TableEmbedding(
            pieces=pieces,
            embedding_dim=embedding_dim,
            structure_length=structure_length,
        )
