import json
import os
import sys
import time
from os import listdir
from os.path import exists, isfile, join
from pathlib import Path
from typing import Optional

import tqdm

import pytorch_lightning as pl
import torch
import typer
from explanatory_learning.data.encoders import get_structure_encoder
from explanatory_learning.data.metrics import soft_guess
from torch.utils.data import DataLoader
from explanatory_learning.learning.datasets import FalsifierDataset, LitFalsifierModel, ZendoSemantics

STRUCTURE_ENCODER = get_structure_encoder()
fals_batch2dict = lambda x: {"rules": x[0], "structures": x[1], "labels": x[2]}


def create_conjecture(model, rule: str, dataset, batch2dict, batch_size=1170):
    rule_id = dataset.id_from_rule(rule)
    labeling = dataset.rule_labeling(rule_id)

    data_loader = DataLoader(
        labeling, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
    )

    conjecture = {}
    device = model.device

    # metrics
    accuracy = pl.metrics.classification.Accuracy(compute_on_step=False).to(device)
    f1 = pl.metrics.classification.F1(num_classes=2, compute_on_step=False).to(device)

    with torch.no_grad():
        starting_time = time.time()
        for batch_idx, batch in enumerate(data_loader):

            batch = [x.to(device) for x in batch]
            model_args = batch2dict(batch)
            labels = model_args["labels"]
            del model_args["labels"]

            # compute loss
            output = model(**model_args)
            if type(output) == tuple:
                predicted_labels = output[0].argmax(dim=-1)
            else:
                predicted_labels = output.argmax(dim=-1)

            # save metrics
            accuracy(preds=predicted_labels, target=labels)
            f1(preds=predicted_labels, target=labels)

            # update current conjecture
            predicted_labels_list = predicted_labels.tolist()

            structures_strings = STRUCTURE_ENCODER.inverse_transform(
                model_args["structures"]
            )
            for i, structure in enumerate(structures_strings):
                conjecture[structure] = predicted_labels_list[i]

    output_data = {
        "conjecture": conjecture,
        "accuracy": accuracy.compute().item(),
        "f1": f1.compute().item(),
        "time": time.time() - starting_time,
    }
    return output_data


def metrics(model, dataset, batch2dict, filename: str = None):
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

        results = create_conjecture(
            model, rule=rule, dataset=dataset, batch2dict=batch2dict
        )

        rule_conjecture = results["conjecture"]
        accuracy = results["accuracy"]
        f1 = results["f1"]
        pred_time = results["time"]

        (is_correct, ham_distance, is_unique) = soft_guess(
            rule_conjecture=rule_conjecture,
            true_rule=rule,
            rule_labels_split=10,
            verbose=False,
            check_uniqueness=True,
        )

        ham_distance = ham_distance.item()

        sum_distance += ham_distance
        sum_softaccuracy_ambiguous += is_correct

        sum_accuracy += accuracy
        sum_f1 += f1

        if is_correct:
            assert is_unique is not None
            if is_unique:
                sum_softaccuracy += is_correct
                sum_distance_correct += ham_distance
            else:
                ambiguous_guesses += not is_unique

    num_rules = len(dataset.rules)
    soft_accuracy = sum_softaccuracy / num_rules
    soft_accuracy_ambiguous = sum_softaccuracy_ambiguous / num_rules

    avg_distance = sum_distance / num_rules
    avg_distance_correct = (
        float("inf")
        if sum_softaccuracy == 0
        else sum_distance_correct / sum_softaccuracy
    )

    avg_f1 = sum_f1 / num_rules
    avg_accuracy = sum_accuracy / num_rules

    result_data = {
        "NRS": soft_accuracy,
        "NRS_with_ambiguous": soft_accuracy_ambiguous,
        "avg-accuracy": avg_accuracy,
        "avg-f1": avg_f1,
        "avg_hamming_distance": avg_distance,
        "avg_hamming_distance_correct": avg_distance_correct,
    }

    if filename is not None:
        with open(filename, "w") as f:
            json.dump(result_data, f)
    return result_data




def get_model_checkpoint(checkpoint_directory: str, type: str = "complete"):
    onlyfiles = [
        f
        for f in listdir(checkpoint_directory)
        if isfile(join(checkpoint_directory, f))
    ]
    onlyfiles = sorted(onlyfiles)

    for file in onlyfiles:
        file_parts = file.split("-")
        if file_parts[0] == type:
            full_filename = join(checkpoint_directory, file)
            return full_filename
    assert False


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
ZendoSemantics.set_device(DEVICE)


def main(
    training_dir: str = join(root_dir, "training_results"),
    output_dir: str = join(root_dir, "interpreter_analysis_results"),
    config_file: Optional[str] = join(root_dir, "default.json"),
):

    if not os.path.exists(training_dir):
        typer.echo(
            message=f"Error: directory {training_dir} does not exists! Run 'training.py' to create!",
            err=True,
        )
        raise typer.Exit()

    # check that output_folder doesn't exists
    if os.path.exists(output_dir):
        typer.echo(message=f"Error: directory {output_dir} exists already!", err=True)
        raise typer.Exit()

    with open(config_file, "r") as f:
        config_data = json.load(f)

    dataset_test = config_data["test_dataset"]
    dataset_sizes = [
        f"{ts['structures']}structures-{ts['rules']}rules"
        for ts in config_data["training_sets"]
    ]
    num_guesses = config_data["num_beams"]

    # test_dataset =  datasets_from_configurations(validation_configurations, SpecialAwareDataset)["complete"]
    dataset_args = {
        "json_file": dataset_test,
        "num_samples": 1176,
        "rule_filter": lambda x: not ("at_the_left_of" in x),
    }
    test_dataset = FalsifierDataset(**dataset_args)

    training_interpreter = join(training_dir, "interpreter")

    os.mkdir(output_dir)
    for size in dataset_sizes:
        filename = join(output_dir, f"{size}.json")

        model_interpreter = get_model_checkpoint(
            join(training_interpreter, f"{size}/checkpoints")
        )
        interpreter = LitFalsifierModel.load_from_checkpoint(model_interpreter)

        interpreter.to(DEVICE)
        interpreter.eval()

        metrics(
            interpreter,
            dataset=test_dataset,
            batch2dict=fals_batch2dict,
            filename=filename,
        )


if __name__ == "__main__":
    import typer

    typer.run(main)
