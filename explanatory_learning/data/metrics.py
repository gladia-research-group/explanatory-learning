import math
from typing import Dict

import torch

from explanatory_learning.data.utils import ZendoSemantics


def soft_guess(
    rule_conjecture: Dict[str, bool],
    true_rule: str,
    rule_labels_split: int = 3,
    verbose: bool = True,
    check_uniqueness: bool = False,
):
    zsemantics = ZendoSemantics.instance()

    I = [zsemantics.id_from_struct(s) for s in rule_conjecture.keys()]
    S = zsemantics.matrix[:, I]
    device = S.device
    rule_conjecture = torch.tensor(
        [rule_conjecture[s] for s in rule_conjecture.keys()],
        device=device,
        dtype=torch.bool,
    )

    id_true_rule = zsemantics.id_from_rule(true_rule)
    chunk_size = int(math.ceil(len(S) / rule_labels_split))

    ham_dist = torch.logical_xor(rule_conjecture, S)
    ham_dist = torch.cat(
        [
            torch.sum(ham_dist[chunk_size * i : chunk_size * (i + 1)], dim=1)
            for i in range(rule_labels_split)
        ]
    )

    top_sims, top_sims_idx = torch.topk(ham_dist, k=1, largest=False, dim=0)
    sim_true_rule = ham_dist[id_true_rule]
    rule_found = sim_true_rule in top_sims

    uniqueness = None
    if rule_found and check_uniqueness:
        hamdist_nearest = top_sims[0]
        indices_nearest = torch.nonzero(ham_dist == hamdist_nearest).view(-1)
        if indices_nearest.shape[0] == 1:
            uniqueness = True
        else:
            nearest_rules = zsemantics.matrix.index_select(dim=0, index=indices_nearest)
            cardinalities_nearest = nearest_rules.sum(dim=1)

            # compute |A| + |B| - 2*|A int B|
            T = (nearest_rules * nearest_rules[0].view(1, -1)).sum(dim=-1)
            T.mul_(-2).add_(cardinalities_nearest).add_(cardinalities_nearest[0])

            # check that |A| + |B| - 2*|A int B| = 0 i.e A = B for all nearest A and B
            uniqueness = T.sum() == 0

    if verbose:
        print(
            f"  closest rule to conjecture {id_true_rule} (number of wrong labels). True rule:"
        )
        print(f"    {sim_true_rule}  {true_rule :<35}")
        print("     " + "-" * 58)
        rule = zsemantics.rule_from_id(top_sims_idx[0])
        sim = int(top_sims[0])
        print(f"     {sim}  {rule :<35}")
        print()

    return rule_found, sim_true_rule, uniqueness  # or sim_true_rule - 1 in top_sims


# def create_conjecture(model:torch.nn.Module, rule:str, dataset, batch_size=1024):
#     rule_id = dataset.id_from_rule(rule)
#     dataloader = DataLoader(
#              dataset.rule_labeling(rule_id),
#             batch_size=batch_size,
#             shuffle=False,
#             collate_fn=dataset.collate_fn)

#     conjecture = {}
#     device = model.device
#     model.eval()
#     struct_encoder = get_structure_encoder()

#     with torch.no_grad():
#         for batch in dataloader:
#             batch = [x.to(device) for x in batch]
#             rules, structures, labels = batch
#             predicted_tags = model(rules, structures).argmax(dim=-1)

#             predicted_tags = predicted_tags.cpu()
#             structures_strings = struct_encoder.inverse_transform(structures)

#             for i, structure in enumerate(structures_strings):
#                 conjecture[structure] = predicted_tags[i].item()
#     return conjecture

# def soft_accuracy(model:torch.nn.Module, dataset:Dataset, verbose:bool=False):
#     sum_corr=0
#     for rule in tqdm(dataset.rules):
#         rule_conjecture = create_conjecture(model=model, rule=rule, dataset=dataset)
#         sum_corr += soft_guess(
#             rule_conjecture=rule_conjecture,
#             true_rule=rule,
#             top_k=1,
#             rule_labels_split=10,
#             verbose=verbose)

#     soft_acc = sum_corr/len(dataset.rules)
#     return soft_acc

# #-------------------------------------------------------------------------------
# def rule_accuracy(model:torch.nn.Module, dataset:Dataset, device:str, verbose:bool=False) -> Dict[str, float]:
#     model.to(device)
#     model.eval()
#     rule_accuracies = dict()
#     for rule in tqdm(dataset.rules):
#         rule_id = dataset.id_from_rule(rule)
#         rule_labeling = dataset.rule_labeling(rule_id)

#         test_loader = DataLoader(
#             rule_labeling,
#             batch_size=512,
#             shuffle=False,
#             collate_fn=dataset.collate_fn)

#         errors = 0
#         for batch in test_loader:
#             batch = [x.to(device) for x in batch]
#             (rules, structures, labels) = batch

#             prediction =  model(rules, structures).argmax(dim=-1)

#             errors += (prediction != labels).sum()

#         rule_accuracy = 1 - torch.true_divide(errors,len(rule_labeling))
#         if verbose:
#             print(f"accuracy for rule {Style.BRIGHT}\"{rule}\"{Style.RESET_ALL}:\n - {rule_accuracy}\n")
#         rule_accuracies[rule] = rule_accuracy
#     return rule_accuracies
