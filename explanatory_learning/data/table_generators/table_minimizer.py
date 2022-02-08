import math
from typing import List

import torch

from explanatory_learning.data.utils import ZendoSemantics

def _inv_prob(label_preserving_rules:torch.BoolTensor, block_size:int=2000):
    N, M = label_preserving_rules.shape
    num_blocks = int(math.ceil(N/block_size))
    if N*M >= 500*1024**2: #<- 500MiB
        acc = label_preserving_rules[:block_size, :].sum(dim=0, dtype=torch.int16)
        for bi in range(1,num_blocks):
            bstart = bi*block_size
            bend = bstart+block_size
            acc.add_(label_preserving_rules[bstart:bend, :].sum(dim=0, dtype=torch.int16))
    else:
        acc = label_preserving_rules.sum(dim=0)
    return acc


def _get_MIS(old_inv_probability, label_preserving_rules, table_mask):
        max_inv_prob = label_preserving_rules.size(0)
        inv_probabilities = _inv_prob(label_preserving_rules) 

        #mask structures already in the table
        inv_probabilities[table_mask] = max_inv_prob+1 #<- always avoid structures that are already in the 
                                                       #   table.

        # take structure that maximizes the likelihood of the correct rule (Most Informative Structure (MIS))
        mis_idx = inv_probabilities.argmin()

        # get the minimum reciprocal probability
        inv_probability = inv_probabilities[mis_idx]

        assert inv_probability != 0 #<- sanity check

        # check if converged
        converged = old_inv_probability == inv_probability #<- this implies that there are no structures more 
                                                           #   informative then current ones (i.e. uniqueness reached)
        return mis_idx, inv_probability, converged


def minimal_table_greedy(
    rule:str, table_size:int = 10, start_structures:List[str]=[]):
    """
    Computes a table of minimal size using a greedy procedure.

    This procedure computes a (possibly unique) :term:`Table` for the rule ``rule``.
    The resulting table is the result of an iterative greedy procedure.
    At each iteration this procedure computes the most informative structure to insert
    in the output table, resulting in tables with (hopefully) minimal table sizes.

    :param rule: zendo rule consistent with output table
    :param table_size: number of elements in table
    :param start_structures: structures already in initial table.
    
    |

    :returns:
    
        - a List of structures with ``table_size`` elements representing the output table
        - a boolean value which indicates whether the output table is a :term:`Unique table` or not.
    """

    semantics = ZendoSemantics.instance()
    label_mat = semantics.matrix
    device = label_mat.device

    labeling = label_mat[semantics.id_from_rule(rule),:]
    starting_structures = torch.tensor(
        [semantics.id_from_structure(structure) for structure in start_structures], 
        dtype=torch.long, device=device)

    num_rules , num_structures = label_mat.shape

    assert len(labeling.shape) == 1
    assert labeling.size(0) == num_structures
    assert labeling.device == device
    assert labeling.dtype == torch.bool

    old_inv_probability = num_rules+1
    table_mask = torch.zeros([num_structures], dtype=torch.bool, device=device)
    all_label_preserving_rules = label_mat.eq(labeling.view(1,num_structures))
    #all_label_preserving_rules = label_mat.logical_xor(labeling.view(1,num_structures)).logical_not_()

    if starting_structures.nelement() != 0:
        assert len(starting_structures.shape) == 1
        assert starting_structures.dtype == torch.long
        assert table_size > starting_structures.size(0) 

        # fill table mask
        table_mask.index_fill_(dim=0, index=starting_structures, value=1)

        start_struct_labels  = all_label_preserving_rules.index_select(dim=1, index=starting_structures)
        rule_cons_mask, _ = start_struct_labels.min(dim=1) # <- 1 if rule consistent w.r.t starting structures 0 otherwise
        
        # prune rules inconsistent with structures
        rules_cons_idx = torch.nonzero(input=rule_cons_mask).view(-1)
        label_preserving_rules = all_label_preserving_rules.index_select(dim=0, index=rules_cons_idx)
        del start_struct_labels, val, rule_cons_mask
    else:
        label_preserving_rules = all_label_preserving_rules

    # ---------------------------------------------------------
    while table_mask.sum() < table_size: #len(table) < table_size:

        mis_idx, inv_probability, converged = _get_MIS(
            old_inv_probability=inv_probability,
            label_preserving_rules=label_preserving_rules,
            table_mask=table_mask)

        if converged: # check if the algorithm converged (i.e. the table was already unique)
            # structures to add to complete the table
            structs_to_add = table_size - table_mask.sum()
            
            # add final structures randomically
            out_of_table = table_mask.logical_not()
            rand_struct_ids = torch.multinomial(
                input=out_of_table.to(torch.float32), 
                num_samples=structs_to_add)

            for i in range(structs_to_add) :
                table_mask[rand_struct_ids[i]] = 1
            del out_of_table # < free resources

        else:
            # reduce consistency matrix (improves performance, but loses rule indices)
            rules_mask = label_preserving_rules[:,mis_idx] #<- get consistency of rules associated to most informative structure
            label_preserving_rules = label_preserving_rules[rules_mask, :]

            # update table
            table_mask[mis_idx] = 1 #set structure as in table
    #-----------------------------------------

    if not converged: #<- if not already unique check if it reached uniqueness in the last iteration
        _,_, converged = _get_MIS(
            old_inv_probability=inv_probability,
            label_preserving_rules=label_preserving_rules,
            table_mask=table_mask)
    uniqueness = converged.item()

    del all_label_preserving_rules # < free resources

    # return table indices 
    #table = torch.tensor(table, dtype=torch.long, device=device)
    table_ids = torch.nonzero(table_mask).view(-1)
    table = [semantics.struct_from_id(struct_id) for struct_id in table_ids]

    return table, uniqueness
