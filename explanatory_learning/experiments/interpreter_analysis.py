import json
import os
import time
from pathlib import Path
from typing import Union, Dict, Any, Callable

import tqdm

import pytorch_lightning as pl
import torch
import typer
from explanatory_learning.data.encoders import get_structure_encoder
from explanatory_learning.data.metrics import soft_guess
from explanatory_learning.experiments.utils import load_model, validate_experiment_input
from explanatory_learning.learning.datasets import FalsifierDataset
from explanatory_learning.learning.models import LitFalsifierModel
from explanatory_learning.utils import ROOT_DIR
from torch.utils.data import DataLoader

STRUCTURE_ENCODER = get_structure_encoder()
interpreter_batch2dict = lambda x: {"rules": x[0], "structures": x[1], "labels": x[2]}


def create_conjecture(
    model: LitFalsifierModel,
    rule: str,
    dataset: FalsifierDataset,
    batch2dict: Callable,
    batch_size: int = 1170,
) -> Dict[str, Any]:

    rule_id = dataset.id_from_rule(rule)
    labeling = dataset.rule_labeling(rule_id)

    data_loader = DataLoader(
        labeling, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
    )

    # metrics]
    device = model.device
    accuracy = pl.metrics.classification.Accuracy(compute_on_step=False).to(device)
    f1 = pl.metrics.classification.F1(num_classes=2, compute_on_step=False).to(device)

    conjecture = {}
    with torch.no_grad():
        starting_time = time.time()
        for batch_idx, batch in enumerate(data_loader):

            batch = [x.to(device) for x in batch]
            model_args = batch2dict(batch)
            labels = model_args["labels"]
            del model_args["labels"]

            # compute loss
            output = model(**model_args)
            if type(output) == tuple: # TODO: unify interface
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
    ambiguous_guesses = 0
    sum_softaccuracy = 0
    sum_softaccuracy_ambiguous = 0
    sum_distance = 0
    sum_distance_correct = 0
    sum_accuracy = 0
    sum_f1 = 0

    for i, rule in enumerate(tqdm.auto.tqdm(dataset.rules)):
        results = create_conjecture(model, rule, dataset, batch2dict)
        rule_conjecture = results["conjecture"]
        accuracy = results["accuracy"]
        f1 = results["f1"]

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


def main(
    training_dir: Path = ROOT_DIR / "training_results",
    output_dir: Path = ROOT_DIR / "interpreter_analysis_results",
    config_file: Path = ROOT_DIR / "configs/default.json",
):
    validate_experiment_input(training_dir, output_dir, config_file)

    # load configuration
    with open(config_file, "r") as f:
        config = json.load(f)

    # get training hyperparams
    dataset_sizes = [
        f"{ts['structures']}structures-{ts['rules']}rules"
        for ts in config["training_sets"]
    ]

    # instantiate dataset
    test_dataset = FalsifierDataset(
        **{
            "json_file": config["test_dataset"],
            "num_samples": 1176,
            "rule_filter": lambda x: not ("at_the_left_of" in x),
        }
    )

    # compute metrics
    os.mkdir(output_dir)
    for size in dataset_sizes:
        interpreter = load_model(
            training_dir / f"interpreter/{size}/checkpoints",
            model_class=LitFalsifierModel,
        )
        metrics(
            interpreter,
            dataset=test_dataset,
            batch2dict=interpreter_batch2dict,
            filename=str(output_dir / f"{size}.json"),
        )


if __name__ == "__main__":
    typer.run(main)
