from os import listdir
import os
from os.path import  join, exists
from pathlib import  Path
from os.path import isfile, join, exists
import time
import json
import sys
from typing import Optional, List, Dict


root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(root_dir)

import tqdm
from colorama import Style, Fore
from torch.utils.data import DataLoader
from explanatory_learning.data.encoders import get_structure_encoder
from explanatory_learning.data.metrics import soft_guess
import pytorch_lightning as pl

from zendo import ZendoSemantics

def get_model_checkpoint(checkpoint_directory:str, type:str="complete"):
    onlyfiles = [f for f in listdir(checkpoint_directory) if isfile(join(checkpoint_directory,f))]
    onlyfiles = sorted(onlyfiles)
    
    for file in onlyfiles:
        file_parts = file.split("-")
        if file_parts[0] == type:
            full_filename = join(checkpoint_directory, file)
            return full_filename
    assert False



def check_equivalence(rule_1:str, rule_2:str):
    semantics = ZendoSemantics.instance()
    if rule_1 not in semantics._id_from_rule:
        #print(f"Warning {rule_1} is not correct!")
        return False
    
    if rule_2 not in semantics._id_from_rule:
        #print(f"Warning {rule_2} is not correct!")
        return False
    
    rule_1_id = semantics.id_from_rule(rule_1)
    rule_2_id = semantics.id_from_rule(rule_2)
    return torch.equal(semantics.matrix[rule_2_id,:], semantics.matrix[rule_1_id,:])



aware_batch2dict = lambda x: {"tables":x[0],"table_labels":x[1], "structures":x[2], "labels":x[3], "rules":x[4]}
from zendo import BeamSearcher
def metrics(num_guesses:int, falsifier, conjecture_generator, empiricist, dataset, filename=None):
    rule_encoder = dataset.rule_encoder
    rationalist = RationalistModel(
        num_beams=num_guesses,
        falsifier=falsifier,
        conjectures_generator=conjecture_generator,
        max_rule_length=dataset._encoded_rules.shape[-1],
        rule_encoder=rule_encoder)
    
    bsearch = BeamSearcher(
            model=empiricist,
            num_beams=num_guesses,
            max_rule_length=13,
            model_args_batch_idx= {"tables":1,"table_labels":1,"structures":1},
            autoregressive_arg="rules",
            rule_encoder=rule_encoder)

    sum_accuracy_rationalist = 0
    sum_accuracy_aware = 0
    sum_accuracy_empiricist = 0
    for i, rule in enumerate(tqdm.auto.tqdm(dataset.rules)):

        rule_id = dataset.id_from_rule(rule)
        idx = dataset.rule_id_to_idx(rule_id)
        batch = dataset.collate_fn([dataset[idx]])
        batch = [x.to(DEVICE) for x in batch]

        (tables,
         table_labels,
         structures_encoded,
         labels,
         rules_encoded) = batch

        rationalist.guess_rule(tables, table_labels, rule, find_nearest=True)

        #if rationalist.hidden_rule is not None:
        predicted_rule_rationalist = rule_encoder.inverse_transform(rationalist.hidden_rule)[0]
        predicted_rule_aware = rule_encoder.inverse_transform(rationalist.most_probable_rule)[0]

        rules_encoded, rule_probabilities = bsearch.sample(tables=tables, table_labels=table_labels)
        rules_encoded = rules_encoded.view(13, num_guesses)
        predicted_rule_empiricist = dataset.rule_encoder.inverse_transform(rules_encoded[:,0:1])[0]

        if check_equivalence(predicted_rule_rationalist, rule) :
            sum_accuracy_rationalist += 1
        
        if check_equivalence(predicted_rule_aware, rule):
            sum_accuracy_aware += 1

        if check_equivalence(predicted_rule_empiricist, rule):
            sum_accuracy_empiricist += 1

    num_rules = len(dataset.rules)
    accuracy_rationalist = sum_accuracy_rationalist/num_rules
    accuracy_aware = sum_accuracy_aware/num_rules
    accuracy_empiricist = sum_accuracy_empiricist/num_rules

    result_data = {"rationalist-accuracy":accuracy_rationalist,
                   "conscious-accuracy":accuracy_aware,
                   "empiricist-accuracy":accuracy_empiricist}
    
    if filename is not None:
        with open(filename, "w") as f: json.dump(result_data, f)
    
    return result_data

import typer
import torch
from zendo import AwareEmpiricistDataset
from zendo import LitAwareEmpiricistModel, LitEmpiricistRuleModel, LitEmpiricistModel, LitFalsifierModel
from explanatory_learning.learning.models.transformers.rationalist import RationalistModel

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
ZendoSemantics.set_device(DEVICE)

def main(
    training_dir : str = join(root_dir, "training_results"),
    output_dir : str = join(root_dir, "rule_induction_results"),
    config_file : Optional[str]=join(root_dir, "default.json")):

    if not os.path.exists(training_dir):
        typer.echo(message=f"Error: directory {training_dir} does not exists! Run 'training.py' to create!", err=True)
        raise typer.Exit()

    # check that output_folder doesn't exists
    if os.path.exists(output_dir):
        typer.echo(message=f"Error: directory {output_dir} exists already!", err=True)
        raise typer.Exit()

    with open(config_file, "r") as f:
        config_data = json.load(f)
    
    dataset_test = config_data["test_dataset"]
    dataset_sizes = [ f"{ts['structures']}structures-{ts['rules']}rules" for ts in config_data["training_sets"]]
    num_guesses = config_data["num_beams"]

#test_dataset =  datasets_from_configurations(validation_configurations, SpecialAwareDataset)["complete"]
    dataset_args = {"json_file":dataset_test, "num_samples": 1, "rule_filter": lambda x: not ("at_the_left_of" in x)}
    test_dataset = AwareEmpiricistDataset(**dataset_args)

    training_conscious =  join(training_dir, "empiricist_conscious")
    training_interpreter = join(training_dir, "interpreter")
    training_empiricist = join(training_dir, "empiricist_rules")

    os.mkdir(output_dir)
    for size in dataset_sizes:
        # load models
        model_conscious = get_model_checkpoint(checkpoint_directory=join(training_conscious, f"{size}/checkpoints"))
        conjecture_generator = LitAwareEmpiricistModel.load_from_checkpoint(model_conscious)

        model_interpreter = get_model_checkpoint(join(training_interpreter, f"{size}/checkpoints"))
        interpreter = LitFalsifierModel.load_from_checkpoint(model_interpreter)

        model_empiricist = get_model_checkpoint(join(training_empiricist, f"{size}/checkpoints"))
        empiricist = LitEmpiricistRuleModel.load_from_checkpoint(model_empiricist)

        interpreter.to(DEVICE)
        interpreter.eval()

        conjecture_generator.to(DEVICE)
        conjecture_generator.eval()

        empiricist.to(DEVICE)
        empiricist.eval()
    
        # compute metrics
        metrics(
            num_guesses=num_guesses,
            empiricist=empiricist,
            falsifier=interpreter,
            conjecture_generator=conjecture_generator,
            dataset=test_dataset,
            filename=join(output_dir, f"{size}.json"))
if __name__ == '__main__':
    typer.run(main)