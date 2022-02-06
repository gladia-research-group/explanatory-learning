from typing import List, Dict, Tuple
import numpy as np


edit_costs = {
    '..' : 0,
    '.A' : 3, 'A.' : 3,
    '.B' : 3, 'B.' : 3,
    '.a' : 3, 'a.' : 3,
    '.b' : 3, 'b.' : 3,
    'AA' : 0,
    'AB' : 1, 'BA' : 1,
    'Aa' : 1, 'aA' : 1,
    'Ab' : 2, 'bA' : 2,
    'BB' : 0, 
    'Ba' : 2, 'aB' : 2,
    'Bb' : 1, 'bB' : 1,
    'aa' : 0,
    'ab' : 1, 'ba' : 1,
    'bb' : 0
}


def wedit_dist(edit_costs, a, b):
    if a != '......':
        while a[0] == '.':  # traslation operation is an invariant
            a += '.'
            a = a[1:]
    if b != '......':
        while b[0] == '.':
            b += '.'
            b = b[1:]
    dist = 0
    for i, char_a in enumerate(a):
        dist += edit_costs[char_a + b[i]]
    return dist


def simplicity_by_voids(structure):
    return structure.count('.')


def simplicity_dict_builder(discrete_simplicity_calculator, ordered_structures):
    '''Computes the complexity of input structures.

    Inputs:
        - (function) discrete_simplicity_calculator is a function taking in 
        input a structure and giving in output an int between 0 and 6. 
        0: min simplicity, 6: max simplicity 
        - (list) ordered strucutures is a list of structures ordered by structureid
    
    Outputs:
        - (dict) structuresid_by_simplicity is a dictionary with int keys from 0 to 6
        and list values containing the structures_id of structures of simplicity=key
    '''
    structuresid_by_simplicity = {}
    for i, structure in enumerate(ordered_structures):
        simplicity = discrete_simplicity_calculator(structure)
        if simplicity not in structuresid_by_simplicity:
            structuresid_by_simplicity[simplicity] = [i]
        else:
            structuresid_by_simplicity[simplicity].append(i)
    return structuresid_by_simplicity


def focus_model(oredered_structures:List[str], ordered_labels:List[bool], 
                structuresid_by_simplicity:Dict[int,str], n_couples:int=5, deterministic:bool=False) -> List[Tuple[str,bool,int]]:
    '''Computes relevant tagged structures that may help in the prediction of the hidden rule.

    Inputs:
        - ordered labels: list of labels ordered by structureid
        - ordered strucutures:  list of structures ordered by structureid
        - structuresid_by_simplicity:  dictionary with int keys from 0 to 6 and list values containing the structures_id of structures of simplicity=key
        - n_couples is the number of couples of structures the user wants in output
        - deterministic: if True this function is deterministic
    
    Outputs:
        - relevant_structures: list containing the n_couples*2 relevant strucutures. Each entry is a tuple containing: (structure, label, idstructure)
    '''
    relevant_structures = []
    new_relevant_structures_should_be_far_from = []
    for j in range(n_couples):
        find = False
        max_dist = -2
        for i in sorted(structuresid_by_simplicity.keys(), reverse=True):
            prevalent_label = ordered_labels[0]  # label of '......'
            if deterministic:
                structures_of_given_simplicity = np.array(structuresid_by_simplicity[i])
            else:
                structures_of_given_simplicity = np.random.permutation(np.array(structuresid_by_simplicity[i]))
            for id_structure in structures_of_given_simplicity:
                if ordered_labels[id_structure] != prevalent_label:
                    dist = -1
                    for structure_you_want_to_stay_away in new_relevant_structures_should_be_far_from:
                        if dist < 0:
                            dist = 1
                        dist *= max(0, wedit_dist(edit_costs, structures[id_structure], structure_you_want_to_stay_away) - 2*(6-i))  # complexity penalty
                    if dist > max_dist:
                        relevant_structure = (structures[id_structure], ordered_labels[id_structure], id_structure)
                        max_dist = dist
                        # print(j, relevant_structure, dist)
                    if dist == -1:
                        find = True
                        break
            if find:
                break
        relevant_structures.append(relevant_structure)
        new_relevant_structures_should_be_far_from.append(relevant_structure[0])

        min_dist = 100
        for i in sorted(structuresid_by_simplicity.keys(), reverse=True):
            if deterministic:
                structures_of_given_simplicity = np.array(structuresid_by_simplicity[i])
            else:
                structures_of_given_simplicity = np.random.permutation(np.array(structuresid_by_simplicity[i]))
            for id_structure in structures_of_given_simplicity:
                if ordered_labels[id_structure] != relevant_structure[1]:  # different label 
                    if wedit_dist(edit_costs, structures[id_structure], relevant_structure[0]) < min_dist:  # take the lowest distance distance 
                        second_relevant_structure = (structures[id_structure], ordered_labels[id_structure], id_structure) 
                        min_dist = wedit_dist(edit_costs, structures[id_structure], relevant_structure[0])
                        # print(j, second_relevant_structure)
        relevant_structures.append(second_relevant_structure)
    return relevant_structures

if __name__ == "__main__": 

    ### Download and unzip example dataset using:
    # !wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RDYHilHg9wExyfJ9_tQ6gxujSusumZMz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RDYHilHg9wExyfJ9_tQ6gxujSusumZMz" -O ZendoDatasetS6.zip && rm -rf /tmp/cookies.txt
    # !7z x "/content/ZendoDatasetS6.zip" -o/content/
    
    examples = [
    ('a.....', 'a.....'),
    ('a.....', '.....a'),
    ('a.....', 'b.....'),
    ('a.....', 'A.....'),
    ('a.....', '...a..'),
    ('a.bb..', '..bb.a'),
    ('a.....', '...b..'),
    ('ab....', '...b.a'),
    ('aAb...', 'b.bB..'),
    ('aAb...', 'AB...A'),
    ('aAb...', '.A.b.B'),
    ('aAb...', '.A.bBB'),
    ('a.....', 'ab....')            
    ]

    print('########## NAIF WEIGHTED EDIT DISTANCE EXAMPLE (structure A, structure B, dist')
    for ex in examples:
        print(ex, wedit_dist(edit_costs, ex[0], ex[1]))

    print('########## SIMPLICITY BY VOIDS EXAMPLE (structure, simplicity)')
    for ex in examples:
        print(ex[1], simplicity_by_voids(ex[1]))

    print('########## FOCUS MODEL EXAMPLE (structure, label, structureid)')
    with open("/content/ZendoDatasetS6/structures.txt") as f:
        structures = f.read().splitlines()
    rule = '02'
    with open('/content/ZendoDatasetS6/00000000' + rule + '/labels.txt') as f:
        labels = f.read().splitlines()
        labels = np.asarray(labels, dtype=np.int)
    sim_dic = simplicity_dict_builder(simplicity_by_voids, structures)
    print('simplicity dict built')
    relevant_structures = focus_model(structures, labels, sim_dic)
    for rel_str in relevant_structures:
        print(rel_str)
    with open('/content/ZendoDatasetS6/00000000' + rule + '/rule.txt') as f:
        print('rule: ', f.read())