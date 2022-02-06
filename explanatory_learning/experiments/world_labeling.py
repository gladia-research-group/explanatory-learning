import os
from pathlib import  Path
import sys
import time
import json
from typing import Optional

import tqdm
from colorama import Style, Fore
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(root_dir)

from explanatory_learning.data.encoders import get_structure_encoder
from explanatory_learning.data.metrics import soft_guess

aware_batch2dict = lambda x: {"tables":x[0],"table_labels":x[1], "structures":x[2], "labels":x[3], "rules":x[4]}
empirist_batch2dict = lambda x:{"tables":x[0],"table_labels":x[1], "structures":x[2], "labels":x[3]}

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
STRUCTURE_ENCODER = get_structure_encoder()

from zendo import ZendoSemantics
ZendoSemantics.set_device(DEVICE)

def create_conjecture_rationalist(rationalist, rule:str, dataset, batch_size=1024):
    rule_id = dataset.id_from_rule(rule)
    labeling = dataset.rule_labeling(rule_id)

    data_loader = DataLoader(
            labeling, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=dataset.collate_fn)

    conjecture = {}
    device = rationalist.conjectures_generator.device
    
    # metrics
    accuracy = pl.metrics.classification.Accuracy(compute_on_step=False).to(device)
    f1 = pl.metrics.classification.F1(num_classes=2, compute_on_step=False).to(device)
    #confusion_matrix = pl.metrics.classification.ConfusionMatrix(num_classes=2, compute_on_step=False).to(device)

    with torch.no_grad():
        starting_time = time.time()
        for batch in data_loader:
            batch = [x.to(device) for x in batch]

            batch_dict = aware_batch2dict(batch)

            structures = batch_dict["structures"]
            labels = batch_dict["labels"]
            
            # compute loss
            predicted_labels = rationalist.predict_labels(structures)

            # save metrics
            accuracy(preds=predicted_labels, target=labels)
            f1(preds=predicted_labels, target=labels)
            #confusion_matrix(preds=predicted_labels, target=labels)
            
            # update current conjecture
            predicted_labels_list = predicted_labels.tolist()
            structures_strings = STRUCTURE_ENCODER.inverse_transform(structures)
            for struct_idx, structure in enumerate(structures_strings):
                conjecture[structure] = predicted_labels_list[struct_idx]

    output_data = {
        "conjecture":conjecture,
        "accuracy":accuracy.compute().item(),
        "f1":f1.compute().item(),
        #"confusion_matrix":confusion_matrix.compute().tolist(),
        "time":time.time() - starting_time
    }
    return output_data

from explanatory_learning.learning.models.transformers.rationalist import RationalistModel
def metrics_rationalist(num_guesses:int, falsifier, conjecture_generator, dataset, filename=None):
    rule_encoder = dataset.rule_encoder
    rationalist = RationalistModel(
        num_beams=num_guesses,
        falsifier=falsifier,
        conjectures_generator=conjecture_generator,
        max_rule_length=dataset._encoded_rules.shape[-1],
        rule_encoder=rule_encoder)

    ambiguous_guesses = 0

    sum_softaccuracy = 0
    sum_softaccuracy_ambiguous = 0
    sum_distance = 0
    sum_distance_correct = 0
    sum_accuracy = 0
    sum_f1 = 0

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

        rationalist.guess_rule(tables, table_labels, rule, find_nearest=True )

        #if rationalist.hidden_rule is not None:
        predicted_rule = rule_encoder.inverse_transform(rationalist.hidden_rule)[0]
        
        results = create_conjecture_rationalist(rationalist, rule=rule, dataset=dataset)
        
        rule_conjecture = results["conjecture"]
        accuracy = results["accuracy"]
        f1 = results["f1"]
        pred_time = results["time"]

        (is_correct,
        ham_distance,
        is_unique) = soft_guess(
            rule_conjecture=rule_conjecture,
            true_rule=rule,
            rule_labels_split=8,
            verbose=False,
            check_uniqueness=True)                
            
        ham_distance = ham_distance.item()
        sum_distance += ham_distance
        sum_softaccuracy_ambiguous += is_correct

        sum_accuracy += accuracy
        sum_f1 += f1

        if is_correct:
            assert is_unique is not None
            if  is_unique:
                sum_softaccuracy += is_correct
                sum_distance_correct += ham_distance
            else:
                ambiguous_guesses += not is_unique

    num_rules = len(dataset.rules)
    soft_accuracy = sum_softaccuracy/num_rules
    soft_accuracy_ambiguous = sum_softaccuracy_ambiguous/num_rules
    
    avg_distance = sum_distance/num_rules
    avg_distance_correct =  float("inf") if sum_softaccuracy == 0 else sum_distance_correct/sum_softaccuracy

    avg_f1 =  sum_f1/num_rules
    avg_accuracy = sum_accuracy/num_rules

    result_data = {
        "soft_accuracy":soft_accuracy,
        "soft_accuracy_with_ambiguous":soft_accuracy_ambiguous,
        "avg-accuracy":avg_accuracy,
        "avg-f1":avg_f1,
        "avg_hamming_distance":avg_distance,
        "avg_hamming_distance_correct":avg_distance_correct
    }
    
    if filename is not None:
        with open(filename, "w") as f: json.dump(result_data, f)
    
    return result_data

def create_conjecture(empirist, rule:str, dataset, batch2dict, batch_size=1024):
    rule_id = dataset.id_from_rule(rule)
    labeling = dataset.rule_labeling(rule_id)

    data_loader = DataLoader(
            labeling, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=dataset.collate_fn)

    conjecture = {}
    device = empirist.device
    
    # metrics
    accuracy = pl.metrics.classification.Accuracy(compute_on_step=False).to(device)
    f1 = pl.metrics.classification.F1(num_classes=2, compute_on_step=False).to(device)
    #confusion_matrix = pl.metrics.classification.ConfusionMatrix(num_classes=2, compute_on_step=False).to(device)
    
    with torch.no_grad():
        starting_time = time.time()
        for batch_idx, batch in enumerate(data_loader):

            batch = [x.to(device) for x in batch]
            model_args = batch2dict(batch)
            labels = model_args["labels"]
            
            del model_args["rules"]
            del model_args["labels"]
            
            # compute loss
            output = empirist(**model_args)
            if type(output) == tuple:
                predicted_labels = output[0].argmax(dim=-1)
            else:
                predicted_labels = output.argmax(dim=-1)
            
            # save metrics
            accuracy(preds=predicted_labels, target=labels)
            f1(preds=predicted_labels, target=labels)
            #confusion_matrix(preds=predicted_labels, target=labels)
            
            # update current conjecture
            predicted_labels_list = predicted_labels.tolist()

            structures_strings = STRUCTURE_ENCODER.inverse_transform(model_args["structures"])
            for i,structure in enumerate(structures_strings):
                conjecture[structure] = predicted_labels_list[i]

    output_data = {
        "conjecture":conjecture,
        "accuracy":accuracy.compute().item(),
        "f1":f1.compute().item(),
        #"confusion_matrix":confusion_matrix.compute().tolist(),
        "time":time.time() - starting_time
    }
    return output_data

def metrics_empiricist(empiricist, dataset, batch2dict, filename:str=None):
    rule_encoder = dataset.rule_encoder

    ambiguous_guesses = 0
    sum_softaccuracy = 0
    sum_softaccuracy_ambiguous = 0
    sum_distance = 0
    sum_distance_correct = 0
    sum_accuracy = 0
    sum_f1 = 0
    for i, rule in enumerate(tqdm.auto.tqdm(dataset.rules)):

        rule_id = dataset.id_from_rule(rule)
        idx = dataset.rule_id_to_idx(rule_id)
        
        results = create_conjecture(empiricist, rule=rule, dataset=dataset, batch2dict=batch2dict)
        
        rule_conjecture = results["conjecture"]
        accuracy = results["accuracy"]
        f1 = results["f1"]
        pred_time = results["time"]

        (is_correct,
        ham_distance,
        is_unique) = soft_guess(
            rule_conjecture=rule_conjecture,
            true_rule=rule,
            rule_labels_split=10,
            verbose=False,
            check_uniqueness=True)                
        
        ham_distance = ham_distance.item()

        sum_distance += ham_distance
        sum_softaccuracy_ambiguous += is_correct

        sum_accuracy += accuracy
        sum_f1 += f1

        if is_correct:
            assert is_unique is not None
            if  is_unique:
                sum_softaccuracy += is_correct
                sum_distance_correct += ham_distance
            else:
                ambiguous_guesses += not is_unique


    num_rules = len(dataset.rules)
    soft_accuracy = sum_softaccuracy/num_rules
    soft_accuracy_ambiguous = sum_softaccuracy_ambiguous/num_rules
    
    avg_distance = sum_distance/num_rules
    avg_distance_correct =  float("inf") if sum_softaccuracy == 0 else sum_distance_correct/sum_softaccuracy

    avg_f1 =  sum_f1/num_rules
    avg_accuracy = sum_accuracy/num_rules

    result_data = {
        "NRS":soft_accuracy,
        "NRS_with_ambiguous":soft_accuracy_ambiguous,
        "avg-accuracy":avg_accuracy,
        "avg-f1":avg_f1,
        "avg_hamming_distance":avg_distance,
        "avg_hamming_distance_correct":avg_distance_correct
    }
    if filename is not None:
        with open(filename, "w") as f: 
            json.dump(result_data, f)
    return result_data


from os import listdir
from os.path import isfile, join, exists
def get_model_checkpoint(checkpoint_directory:str, type:str="complete"):
    onlyfiles = [f for f in listdir(checkpoint_directory) if isfile(join(checkpoint_directory,f))]
    onlyfiles = sorted(onlyfiles)
    
    for file in onlyfiles:
        file_parts = file.split("-")
        if file_parts[0] == type:
            full_filename = join(checkpoint_directory, file)
            return full_filename
    assert False

import typer
import torch
from zendo import AwareEmpiricistDataset
from zendo import LitAwareEmpiricistModel, LitEmpiricistRuleModel, LitEmpiricistModel, LitFalsifierModel

def main(
    training_dir : str = join(root_dir, "training_results"),
    output_dir : str = join(root_dir, "world_labeling_results"),
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
    dataset_args = {"json_file":dataset_test, "num_samples": 1176, "rule_filter": lambda x: not ("at_the_left_of" in x)}
    test_dataset = AwareEmpiricistDataset(**dataset_args)

    training_conscious =  join(training_dir, "empiricist_conscious")
    training_interpreter = join(training_dir, "interpreter")
    training_empiricist = join(training_dir, "empiricist_labels")

    os.mkdir(output_dir)
    for size in dataset_sizes:
        # load models
        model_conscious = get_model_checkpoint(checkpoint_directory=join(training_conscious, f"{size}/checkpoints"))
        conjecture_generator = LitAwareEmpiricistModel.load_from_checkpoint(model_conscious)

        model_interpreter = get_model_checkpoint(join(training_interpreter, f"{size}/checkpoints"))
        interpreter = LitFalsifierModel.load_from_checkpoint(model_interpreter)

        model_empiricist = get_model_checkpoint(join(training_empiricist, f"{size}/checkpoints"))
        empiricist = LitEmpiricistModel.load_from_checkpoint(model_empiricist)

        # print(model_interpreter)
        # print(model_conscious)
        # print(model_empiricist)

        interpreter.to(DEVICE)
        interpreter.eval()

        conjecture_generator.to(DEVICE)
        conjecture_generator.eval()

        empiricist.to(DEVICE)
        empiricist.eval()

        # compute metrics
        metrics_rationalist(
            num_guesses=num_guesses,
            falsifier=interpreter,
            conjecture_generator=conjecture_generator,
            dataset=test_dataset,
            filename=join(output_dir, f"rationalist-{size}.json"))

        metrics_empiricist(conjecture_generator,
                    dataset=test_dataset,
                    batch2dict=aware_batch2dict,
                    filename=join(output_dir, f"conscious-{size}.json"))

        metrics_empiricist(empiricist,
                    dataset=test_dataset,
                    batch2dict=aware_batch2dict,
                    filename=join(output_dir, f"empiricist_labels-{size}.json"))

if __name__ == '__main__':
    typer.run(main)