import json
import os
import time
from pathlib import Path
from typing import Union, Dict

import tqdm

import pytorch_lightning as pl
import torch
import typer
from explanatory_learning.data.encoders import get_structure_encoder
from explanatory_learning.data.metrics import soft_guess
from explanatory_learning.experiments.utils import (
    DEVICE,
    validate_experiment_input,
    load_model,
)
from explanatory_learning.learning.datasets import AwareEmpiricistDataset
from explanatory_learning.learning.models import (
    LitAwareEmpiricistModel,
    LitEmpiricistModel,
    LitFalsifierModel,
)
from explanatory_learning.learning.models.transformers.rationalist import (
    RationalistModel,
)
from torch.utils.data import DataLoader

from explanatory_learning.utils import ROOT_DIR

aware_batch2dict = lambda x: {
    "tables": x[0],
    "table_labels": x[1],
    "structures": x[2],
    "labels": x[3],
    "rules": x[4],
}

STRUCTURE_ENCODER = get_structure_encoder()


def create_conjecture_rationalist(rationalist, rule: str, dataset, batch_size=1024):
    rule_id = dataset.id_from_rule(rule)
    labeling = dataset.rule_labeling(rule_id)

    data_loader = DataLoader(
        labeling, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
    )

    conjecture = {}
    device = rationalist.conjectures_generator.device

    # metrics
    accuracy = pl.metrics.classification.Accuracy(compute_on_step=False).to(device)
    f1 = pl.metrics.classification.F1(num_classes=2, compute_on_step=False).to(device)

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

            # update current conjecture
            predicted_labels_list = predicted_labels.tolist()
            structures_strings = STRUCTURE_ENCODER.inverse_transform(structures)
            for struct_idx, structure in enumerate(structures_strings):
                conjecture[structure] = predicted_labels_list[struct_idx]

    output_data = {
        "conjecture": conjecture,
        "accuracy": accuracy.compute().item(),
        "f1": f1.compute().item(),
        "time": time.time() - starting_time,
    }
    return output_data


def metrics_rationalist(
    num_guesses: int, falsifier, conjecture_generator, dataset, filename=None
):
    rule_encoder = dataset.rule_encoder
    rationalist = RationalistModel(
        num_beams=num_guesses,
        falsifier=falsifier,
        conjectures_generator=conjecture_generator,
        max_rule_length=dataset._encoded_rules.shape[-1],
        rule_encoder=rule_encoder,
    )

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

        (tables, table_labels, structures_encoded, labels, rules_encoded) = batch

        rationalist.guess_rule(tables, table_labels, rule, find_nearest=True)

        # if rationalist.hidden_rule is not None:
        predicted_rule = rule_encoder.inverse_transform(rationalist.hidden_rule)[0]

        results = create_conjecture_rationalist(rationalist, rule=rule, dataset=dataset)

        rule_conjecture = results["conjecture"]
        accuracy = results["accuracy"]
        f1 = results["f1"]
        pred_time = results["time"]

        (is_correct, ham_distance, is_unique) = soft_guess(
            rule_conjecture=rule_conjecture,
            true_rule=rule,
            rule_labels_split=8,
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
        "soft_accuracy": soft_accuracy,
        "soft_accuracy_with_ambiguous": soft_accuracy_ambiguous,
        "avg-accuracy": avg_accuracy,
        "avg-f1": avg_f1,
        "avg_hamming_distance": avg_distance,
        "avg_hamming_distance_correct": avg_distance_correct,
    }

    if filename is not None:
        with open(filename, "w") as f:
            json.dump(result_data, f)

    return result_data


def create_conjecture(
    empirist: Union[LitAwareEmpiricistModel, LitEmpiricistModel],
    rule: str,
    dataset: AwareEmpiricistDataset,
    batch2dict: Dict,
    batch_size: int = 1024,
):
    rule_id = dataset.id_from_rule(rule)
    labeling = dataset.rule_labeling(rule_id)

    data_loader = DataLoader(
        labeling, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
    )

    conjecture = {}
    device = empirist.device

    # metrics
    accuracy = pl.metrics.classification.Accuracy(compute_on_step=False).to(device)
    f1 = pl.metrics.classification.F1(num_classes=2, compute_on_step=False).to(device)

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
        # "confusion_matrix":confusion_matrix.compute().tolist(),
        "time": time.time() - starting_time,
    }
    return output_data


def metrics_empiricist(
    empiricist: Union[LitAwareEmpiricistModel, LitEmpiricistModel],
    dataset: AwareEmpiricistDataset,
    batch2dict: Dict,
    filename: Union[str, Path] = None,
):

    ambiguous_guesses = 0
    sum_softaccuracy = 0
    sum_softaccuracy_ambiguous = 0
    sum_distance = 0
    sum_distance_correct = 0
    sum_accuracy = 0
    sum_f1 = 0
    for i, rule in enumerate(tqdm.auto.tqdm(dataset.rules)):

        results = create_conjecture(empiricist, rule, dataset, batch2dict)
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

    # store results
    result_data = {
        "NRS": soft_accuracy,
        "NRS_with_ambiguous": soft_accuracy_ambiguous,
        "avg-accuracy": avg_accuracy,
        "avg-f1": avg_f1,
        "avg_hamming_distance": avg_distance,
        "avg_hamming_distance_correct": avg_distance_correct,
    }
    if filename is not None:
        with open(str(filename), "w") as f:
            json.dump(result_data, f)
    return result_data


def main(
    training_dir: Path = ROOT_DIR / "training_results",
    output_dir: Path = ROOT_DIR / "world_labeling_results",
    config_file: Path = ROOT_DIR / "configs/default.json",
):
    validate_experiment_input(training_dir, output_dir, config_file)

    with open(config_file, "r") as f:
        config = json.load(f)

    dataset_sizes = [
        f"{ts['structures']}structures-{ts['rules']}rules"
        for ts in config["training_sets"]
    ]

    test_dataset = AwareEmpiricistDataset(
        **{
            "json_file": config["test_dataset"],
            "num_samples": 1176,
            "rule_filter": lambda x: not ("at_the_left_of" in x),
        }
    )

    training_conscious = training_dir / "empiricist_conscious"
    training_interpreter = training_dir / "interpreter"
    training_empiricist = training_dir / "empiricist_labels"

    os.mkdir(output_dir)
    for size in dataset_sizes:
        # load models
        conjecture_generator = load_model(
            training_conscious / f"{size}/checkpoints",
            model_class=LitAwareEmpiricistModel,
        )

        interpreter = load_model(
            training_interpreter / f"{size}/checkpoints",
            model_class=LitFalsifierModel,
        )

        empiricist = load_model(
            training_empiricist / f"{size}/checkpoints",
            model_class=LitEmpiricistModel,
        )

        # compute metrics
        metrics_rationalist(
            num_guesses=config["num_beams"],
            falsifier=interpreter,
            conjecture_generator=conjecture_generator,
            dataset=test_dataset,
            filename=output_dir / f"rationalist-{size}.json",
        )

        metrics_empiricist(
            conjecture_generator,
            dataset=test_dataset,
            batch2dict=aware_batch2dict,
            filename=output_dir / f"conscious-{size}.json",
        )

        metrics_empiricist(
            empiricist,
            dataset=test_dataset,
            batch2dict=aware_batch2dict,
            filename=output_dir / f"empiricist_labels-{size}.json",
        )


if __name__ == "__main__":
    typer.run(main)
