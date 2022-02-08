# -----------------------------------------------------------------------------
# A simple rule compiler and evaluator that accepts the given grammar
#
# Built using python PLY
# https://github.com/dabeaz/ply
#
# """
# Rule grammar:
#     RULE     -> GPROP 'and' GPROP | GPROP 'or' GPROP | GPROP
#     GPROP    -> 'not' PROP | PROP
#     PROP     -> QUANTITY OBJ | QUANTITY OBJ REL OBJ
#     REL      -> 'touching' | 'surrounded_by' | 'at_the_right_of' | 'at_the_left_of'
#     QUANTITY -> 'zero' | 'exactly' NUM | 'at_least' NUM | 'at_most' NUM
#     NUM      -> '1' | '2' | '3'
#     OBJ      -> GROUP | PIECE
#     GROUP    -> 'group_of' DIM PIECE
#     DIM      -> '2' | '3'
#     PIECE    -> SHAPE | COLOR | COLOR SHAPE | VOID
#     COLOR    -> 'red' | 'blue'
#     SHAPE    -> 'pyramid' | 'block'
#     VOID     -> '.'
#
# Expression grammar:
#     ^[aAbB\.]{6}$
#
# Expression recap:
#     red => uppercase
#     blu => lowercase
#     pyramid => {a, A}
#     block => {b, B}
# """
# -----------------------------------------------------------------------------


from typing import List

import ply.lex as lex
import ply.yacc as yacc

# Not used in this file, but used elsewhere (e.g. nltk generator)
grammar = """
RULE         ->     PROP_SIMPLE | PROP | PROP_SIMPLE 'and' PROP_SIMPLE | PROP_SIMPLE 'or' PROP_SIMPLE
PROP         ->     QUANTITY OBJ REL OBJ
PROP_SIMPLE  ->     QUANTITY OBJ
REL          ->     'touching' | 'surrounded_by' | 'at_the_right_of' | 'at_the_left_of'
QUANTITY     ->     'at_least' NUM | 'exactly' NUM | 'at_most' NUM | 'zero'
NUM          ->     '1' | '2'
OBJ          ->     COLOR | SHAPE | COLOR SHAPE
COLOR        ->     'red' | 'blue'
SHAPE        ->     'block' | 'pyramid' | 'pyramid' ORIENTATION
ORIENTATION  ->     'pointing_up' | 'pointing_down'
"""

# -> BUILD THE LEXER

tokens = (
    "AND",
    "OR",
    "NOT",
    "TOUCHING",
    "SURROUNDED_BY",
    "AT_THE_RIGHT_OF",
    "AT_THE_LEFT_OF",
    "ZERO",
    "EXACTLY",
    "AT_LEAST",
    "AT_MOST",
    "1",
    "2",
    "RED",
    "BLUE",
    "PYRAMID",
    "BLOCK",
    "POINTING_UP",
    "POINTING_DOWN"
)

# Tokens
t_AND = r"and"
t_OR = r"or"
t_NOT = r"not"
t_TOUCHING = r"touching"
t_SURROUNDED_BY = "surrounded_by"
t_AT_THE_RIGHT_OF = "at_the_right_of"
t_AT_THE_LEFT_OF = "at_the_left_of"

t_ZERO = "zero"
t_EXACTLY = "exactly"
t_AT_LEAST = "at_least"
t_AT_MOST = "at_most"
t_1 = "1"
t_2 = "2"
t_RED = "red"
t_BLUE = "blue"
t_PYRAMID = "pyramid"
t_BLOCK = "block"
t_POINTING_UP = "pointing_up"
t_POINTING_DOWN = "pointing_down"
# t_VOID = "\\."


# Ignored characters
t_ignore = " \t"


def t_newline(t: lex.Token):
    r"\n+"
    t.lexer.lineno += t.value.count("\n")


def t_error(t: lex.Token):
    print(f"Illegal character {t.value[0]!r}")
    raise ValueError("Rule not valid!")
    # t.lexer.skip(1)


# -> BUILD THE AST COMPILER
# It produces an AST

human_readable = {
    "red": {"A", "B","V"},
    "blue": {"a", "b", "v"},
    "pyramid": {"a", "A","v", "V"},
    "block": {"b", "B"},
    "pointing_up":{"a","A"},
    "pointing_down":{"v", "V"}
}

precedence = (
    ("left", "AND", "OR"),
    ("right", "NOT"),
    ("left", "TOUCHING", "SURROUNDED_BY", "AT_THE_RIGHT_OF", "AT_THE_LEFT_OF"),
    ("right", "AT_LEAST", "AT_MOST", "EXACTLY"),
    ("right", "POINTING_UP", "POINTING_DOWN")
)
names = {}


class Node:
    def __init__(
        self, name: str, children: List["Node"] = None, literal: "Node" = None
    ):
        self.name = name
        if children:
            self.children = children
        else:
            self.children = []
        self.literal = literal

    def __repr__(self) -> str:
        if self.literal is not None:
            return f"{self.name} <{self.literal}>"
        else:
            return f"{self.name}"


def p_rule(p: yacc.YaccProduction):
    """rule : prop_simple AND prop_simple 
            | prop_simple OR prop_simple"""

    p[0] = Node(name="rule", children=[p[1], p[3]], literal=p[2])


def p_rule_simple(p: yacc.YaccProduction):
    """rule : prop
            | prop_simple
    """
    p[0] = Node(name="rule", children=[p[1]])


def p_prop_simple(p: yacc.YaccProduction):
    """ prop_simple : quantity obj
    """
    p[0] = Node(name="prop_simple", children=[p[1], p[2]])


def p_prop_complex(p: yacc.YaccProduction):
    """ prop : quantity obj rel obj
    """
    p[0] = Node(name="prop", children=[p[1], p[2], p[3], p[4]])

def p_rel(p: yacc.YaccProduction):
    """ rel : TOUCHING
            | SURROUNDED_BY
            | AT_THE_RIGHT_OF
            | AT_THE_LEFT_OF
    """
    p[0] = Node(name="rel", literal=p[1])

def p_quantity_zero(p: yacc.YaccProduction):
    """ quantity : ZERO
    """
    p[0] = Node(name="quantity", literal=p[1])


def p_quantity(p: yacc.YaccProduction):
    """ quantity : EXACTLY num
                 | AT_LEAST num
                 | AT_MOST num
    """
    p[0] = Node(name="quantity", children=[p[2]], literal=p[1])


def p_num(p: yacc.YaccProduction):
    """ num : 1
            | 2
    """
    p[0] = Node(name="num", literal=p[1])


def p_obj(p: yacc.YaccProduction):
    """ obj : shape
              | color
    """
    p[0] = Node(name="obj", children=[p[1]])


def p_obj_color_and_shape(p: yacc.YaccProduction):
    """ obj :  color shape
    """
    p[0] = Node(name="obj", children=[p[1], p[2]])


def p_color(p: yacc.YaccProduction):
    """ color : RED
              | BLUE
    """
    p[0] = Node(name="color", literal=p[1])


def p_shape(p: yacc.YaccProduction):
    """ shape : PYRAMID
              | BLOCK
    """
    p[0] = Node(name="shape", literal=p[1])

def p_shape_orientation(p: yacc.YaccProduction):
    """ shape : PYRAMID orientation
    """
    p[0] = Node(name="shape", children=[p[2]], literal=p[1])



def p_orientation(p: yacc.YaccProduction):
    """ orientation : POINTING_UP
                    | POINTING_DOWN
    """
    p[0] = Node(name="shape", literal=p[1])

def p_error(p: yacc.YaccProduction):
    raise ValueError(f"Rule is not Valid!")


def get_ast_tree(node: Node, _prefix="", _last=True) -> str:
    """ Build a string representation of the AST for the current rule

    :param node: The root the tree to display
    :param _prefix: the current prefix (indentation level)
    :param _last: True if last child
    :return: a string representation of the AST
    """
    tree = f'{_prefix}{"╰╴" if _last else "├╴"}{node}'
    _prefix += "  " if _last else "│ "
    child_count = len(node.children)
    for i, child in enumerate(node.children):
        _last = i == (child_count - 1)
        tree = f"{tree}\n{get_ast_tree(child, _prefix, _last)}"
    return tree

lexer = lex.lex()
parser = yacc.yacc()


