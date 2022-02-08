from itertools import product
from typing import List, Optional
from warnings import warn

from colorama import Fore, Style
from nltk import CFG, Nonterminal
from nltk.parse.generate import generate
from tqdm.auto import tqdm

import torch
from explanatory_learning.data.encoders import (BLOCK_IDX, BLUE_IDX,
                                                PIECE_EMBEDDING_SIZE,
                                                POINTDOWN_IDX, POINTUP_IDX,
                                                PYRAMID_IDX, RED_IDX,
                                                encode_structure)
from explanatory_learning.data.rule_grammar import Node, parser, grammar
from torch import BoolTensor

_blue_block = f"{Fore.BLUE}\u25A0{Style.RESET_ALL}"
_red_block = f"{Fore.RED}\u25A0{Style.RESET_ALL}"
_blue_pyramid_up = f"{Fore.BLUE}\u25B2{Style.RESET_ALL}"
_red_pyramid_up = f"{Fore.RED}\u25B2{Style.RESET_ALL}"
_blue_pyramid_down = f"{Fore.BLUE}\u25BC{Style.RESET_ALL}"
_red_pyramid_down = f"{Fore.RED}\u25BC{Style.RESET_ALL}"

_char2shape = {
    ".": "_",
    "a": _blue_pyramid_up,
    "A": _red_pyramid_up,
    "b": _blue_block,
    "B": _red_block,
    "v": _blue_pyramid_down,
    "V": _red_pyramid_down,
}
_idx2label = [f"\u25CF", f"\u25CB"]


def pprint_piece(piece: str):
    """Pretty print string representation of input piece using colored unicode geometric shapes."""
    assert len(piece) == 1
    assert piece in _char2shape.keys()
    return _char2shape[piece]


def pprint_structure(structure: str):
    """Pretty print string representation of input structure using colored unicode geometric shapes."""
    return " ".join([pprint_piece(piece) for piece in structure])


def pprint_label(label: bool):
    """Pretty print string representation of input label using colored unicode geometric shapes."""
    idx = int(label)
    assert idx in {0, 1}
    return _idx2label[idx]


# ------------------------------------------------------------------------------


class ZendoSemantics(object):
    """Singleton class wrapping the Zendo semantics.

    This class stores a boolean matrix  :math:`S` of size [num. rules, num. structures] containing the tag of
    each structure with respect of the rule (i.e. :math:`S[r,s] = 1` iff the structure :math:`s` has white tag for the rule :math:`r`).

    .. note::

        This class implements the singletone design pattern, thus use the static method ``instance`` in order to access it.

    Example:

        >>> inst = ZendoSemantics.instance(device="cuda:0")
        >>> rule = "exactly 2 blue square"
        >>> rule_id = inst.id_from_rule(rule)
        >>> print(inst.matrix[rule_id,:])
        tensor([0, 0, 0,  ..., 1, 0, 0], device='cuda:0')
    """

    _device = "cpu"
    _instance = None

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = torch.device("cpu")
        else:
            if type(device) == str:
                device = torch.device(device)

        self._rules = generate_rules()  # < list of all possible rules
        self._structures = generate_structures()  # < list of all possible structures

        self._id_from_rule = {rule: i for i, rule in enumerate(self.rules)}
        self._id_from_struct = {struct: i for i, struct in enumerate(self.structures)}

        tmp = torch.tensor(
            list(map(encode_structure, self.structures)),
            device=device,
            dtype=torch.bool,
        )
        self._structure_enc = tmp.view([self.num_structures, -1, PIECE_EMBEDDING_SIZE])
        self._rules_ast = [parser.parse(rule) for rule in self.rules]
        self.device = (
            tmp.device
        )  # <- necessary since the current GPU is not known until instantiation

        self._matrix = torch.zeros(
            self.num_rules, self.num_structures, dtype=torch.bool, device=device
        )
        for idx, rule_ast in enumerate(tqdm(self._rules_ast)):
            self._matrix[idx, :] = evaluate_encoded(rule_ast, self._structure_enc)

    @staticmethod
    def set_device(device: str = "cpu"):
        """
        :param device: device in which to allocate [or move] the semantics matrix.
        """
        ZendoSemantics._device = device
        if ZendoSemantics._instance is not None:
            ZendoSemantics.instance().to(device)

    @staticmethod
    def instance():
        """Returns the singleton instance of this class.

        :return: singleton instance of this class.

        :Example:
                >>> instance = ZendoSemantics.instance(device="cuda:0")
                >>> print(instance.matrix)
                torch.Size([25000, 117000])
        """
        if ZendoSemantics._instance is None:
            ZendoSemantics._instance = ZendoSemantics(device=ZendoSemantics._device)
        return ZendoSemantics._instance

    @property
    def rules(self) -> List[str]:
        """List of all possible rules."""
        return self._rules

    @property
    def structures(self) -> List[str]:
        """List of all possible structures."""
        return self._structures

    @property
    def num_rules(self) -> int:
        """Number of all possible rules.

        This value is equal to the number of rows in the semantics matrix.
        """
        return len(self.rules)

    @property
    def num_structures(self) -> int:
        """Number of all possible structures.

        This value is equal to the number of columns in the semantics matrix.
        """
        return len(self.structures)

    @property
    def matrix(self) -> torch.Tensor:
        """Matrix containing the tags for each (rule, structure) pair.

        This property returns the boolean matrix of size [num. rules, num.structures]
        containing in each row the tags of all possible structures, and in each column
        the tags of all possible rules.

        """
        return self._matrix

    def id_from_rule(self, rule: str) -> int:
        """Converts a rule to the corresponding row-index.

        :param rule: rule to convert.

        :return: rule-id of the input rule ``rule``.

        """
        try:
            return self._id_from_rule[rule]
        except KeyError as e:
            raise ValueError(f'Rule "{rule}" is not valid!')

    def id_from_struct(self, struct: str) -> int:
        """Converts a structure to the corresponding column-index.

        :param struct: structure to convert.

        :return: structure-id of input structure ``struct``.

        """
        try:
            return self._id_from_struct[struct]
        except KeyError as e:
            raise ValueError(f'Structure "{struct}" is not valid!')

    def rule_from_id(self, rule_id: int) -> str:
        """Converts a rule index to the corresponding rule.

        :param rule_id: id of the rule to retrieve.

        :return: rule associated to the rule-id ``rule_id``.

        """
        return self.rules[rule_id]

    def struct_from_id(self, struct_id: int) -> str:
        """Converts a structure index to the corresponding structure.

        :param struct_id: id of the structure to retrieve.

        :return: structure associated to the structure-id ``struct_id``.

        """
        return self.structures[struct_id]

    def to(self, device: str):
        """Moves the semantic matrix to the desired device.

        :param device: string representation of the target device.
        """
        tmp = self._matrix.to(device=device)
        device_src = self._matrix.device
        device_tgt = tmp.device
        if device_tgt != device_src:
            warn("Moved semantic matrix from {device_src} to {device_tgt}!")
            del self._matrix
            self._matrix = tmp

# -------------------------------------------------------------------------------------------------------------


def evaluate_encoded(ast: Node, structures: torch.BoolTensor) -> torch.BoolTensor:
    """
    Evaluate a set of encoded structure against a rule.

    The rule is passed as an Abstract Syntax Tree (AST) to the function. The
    encoded structures are given as a matrix with size [N, M, E], with N the
    number of structures, and M the number of elements in the structure, and
    E the size of the piece encoding.

    :param ast: the AST of a rule
    :param structures: boolean matrix of size N x M containing the encoded structures

    :return: boolean vector of size N, containing the truth value of the
        input rule for each structure.
    """
    assert len(structures.size()) == 3
    assert structures.size(2) == PIECE_EMBEDDING_SIZE

    if ast.name == "rule":
        return _evaluate_rule_encoded(ast, structures)
    elif ast.name == "prop":
        return _evaluate_prop_encoded(ast, structures)
    elif ast.name == "prop_simple":
        return _evaluate_prop_simple_encoded(ast, structures)
    elif ast.name == "obj":
        return _evaluate_obj_encoded(ast, structures)
    elif ast.name == "color":
        return _evaluate_color_encoded(ast, structures)
    elif ast.name == "shape":
        return _evaluate_shape_encoded(ast, structures)
    assert False


def _evaluate_rule_encoded(node: Node, structures: torch.BoolTensor):
    if node.literal is None and len(node.children) == 1:
        return evaluate_encoded(node.children[0], structures)

    if node.literal == "or":
        p1 = evaluate_encoded(node.children[0], structures)
        p2 = evaluate_encoded(node.children[1], structures)
        return torch.logical_or(p1, p2)

    if node.literal == "and":
        p1 = evaluate_encoded(node.children[0], structures)
        p2 = evaluate_encoded(node.children[1], structures)
        return torch.logical_and(p1, p2)
    assert False


def _evaluate_prop_encoded(node: Node, structures: BoolTensor):
    quantity = _quantity(node.children[0])
    obj_left = evaluate_encoded(node.children[1], structures)
    rel = _rel(node.children[2])
    obj_right = evaluate_encoded(node.children[3], structures)
    rel_objs = rel(obj_left, obj_right)  # <-- vector Nx6
    return quantity(rel_objs)  # <-- vector Nx1


def _evaluate_prop_simple_encoded(node: Node, structures: str):
    quantity = _quantity(node.children[0])
    objects = evaluate_encoded(node.children[1], structures)
    return quantity(objects)  # <- return a vector [1,N]


def _evaluate_obj_encoded(node: Node, structures: torch.Tensor):
    if len(node.children) == 1:
        return evaluate_encoded(node.children[0], structures)

    if len(node.children) == 2:
        color = evaluate_encoded(node.children[0], structures)
        shape = evaluate_encoded(node.children[1], structures)
        return color.logical_and(shape)
    assert False


def _evaluate_color_encoded(node: Node, structures: torch.Tensor):
    N, M, E = structures.shape
    if node.literal == "blue":
        color_idx = BLUE_IDX
    if node.literal == "red":
        color_idx = RED_IDX
    structures = structures[:, :, color_idx]
    structures = structures.reshape(N, M)
    return structures  # <- size N x 6


def _evaluate_shape_encoded(node: Node, structures: torch.Tensor):
    N, M, E = structures.shape

    if node.literal == "pyramid":
        shape_idx = PYRAMID_IDX
    if node.literal == "block":
        shape_idx = BLOCK_IDX
    shape = structures[:, :, shape_idx].reshape(N, M)

    if len(node.children) == 1:
        pointing_idx = _point(node.children[0])
        pointing = structures[:, :, pointing_idx].reshape(N, M)
        shape = shape.logical_and(pointing)
    return shape  # <- size N x M


def _point(node: Node):
    if node.literal == "pointing_up":
        return POINTUP_IDX
    elif node.literal == "pointing_down":
        return POINTDOWN_IDX
    assert False


def _num(node: Node):
    return int(node.literal)


def _quantity_fn(x, num, op: str):
    assert len(x.shape) == 2
    sums = x.sum(dim=1)
    if op == "exactly":
        return sums == num
    if op == "at_least":
        return sums >= num
    if op == "at_most":
        return sums <= num
    if op == "zero":
        return sums == 0
    assert False


def _quantity(node: Node):
    num = _num(node.children[0]) if len(node.children) > 0 else 0
    op = node.literal
    return lambda x: _quantity_fn(x=x, num=num, op=node.literal)


def _rel(node: Node):
    rel = node.literal
    if rel == "surrounded_by":
        return lambda x, y: _touching_left(x, y).logical_and(_touching_right(x, y))
    if rel == "touching":
        return lambda x, y: _touching_left(x, y).logical_or(_touching_right(x, y))
    if rel == "at_the_right_of":
        return _right_of
    if rel == "at_the_left_of":
        return _left_of
    assert False


def _left_of(objs1, objs2):
    N, M = objs1.shape
    res = torch.zeros([N, M], dtype=torch.bool, device=objs1.device)
    obj_shifted = objs2
    for i in range(M - 1):
        obj_shifted = obj_shifted.roll(-1, dims=1)
        obj_shifted[:, -1] = 0
        tmp = objs1.logical_and(obj_shifted)  # <- NxM
        res.logical_or_(tmp)
    return res


def _right_of(objs1, objs2):
    N, M = objs1.shape
    res = torch.zeros_like(objs1, dtype=torch.bool, device=objs1.device)
    obj_shifted = objs2
    for i in range(M - 1):
        obj_shifted = obj_shifted.roll(1, dims=1)
        obj_shifted[:, 0] = 0
        tmp = objs1.logical_and(obj_shifted)  # <- NxM
        res.logical_or_(tmp)
    return res


def _touching_right(objs1, objs2):
    obj_shifted = objs2.roll(1, dims=1)
    obj_shifted[:, 0] = 0
    obj_shifted.logical_and_(objs1)
    return obj_shifted  # <- NxM


def _touching_left(objs1, objs2):
    obj_shifted = objs2.roll(-1, dims=1)
    obj_shifted[:, -1] = 0
    obj_shifted.logical_and_(objs1)
    return obj_shifted  # <- NxM

# -----------------------------------------------------------------------------


def generate_rules(
    max_rule_depth: Optional[int] = 6,
    max_n: Optional[int] = None,
    start_nonterminal: Optional[Nonterminal] = None,
) -> List[str]:
    """
    Generates all possible rules.

    :param max_rule_depth: maximum rule depth
    :param max_n: maximum number of rules to generate
    :param start_nonterminal: nonterminal to use as start
    :return: list of generated rules
    """
    parsed_grammar = CFG.fromstring(grammar)
    rules = sorted(
        " ".join(sentence)
        for sentence in generate(
            parsed_grammar, depth=max_rule_depth, n=max_n, start=start_nonterminal
        )
    )
    return rules


def generate_structures() -> List[str]:
    """
    Generates all possible Zendo structures.

    :return: list of generated structures
    """
    return sorted("".join(x) for x in product("aAbBvV.", repeat=6))
