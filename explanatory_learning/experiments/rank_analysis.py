import os
import json
from typing import Optional

import tqdm
from torch.utils.data import DataLoader

from zendo import BeamSearcher
from zendo import ZendoSemantics


def check_equivalence(rule_1: str, rule_2: str):  # TODO: specify error behavior
    semantics = ZendoSemantics.instance()
    if rule_1 not in semantics._id_from_rule:
        # print(f"Warning {rule_1} is not correct!")
        return False

    if rule_2 not in semantics._id_from_rule:
        # print(f"Warning {rule_2} is not correct!")
        return False

    rule_1_id = semantics.id_from_rule(rule_1)
    rule_2_id = semantics.id_from_rule(rule_2)
    return torch.equal(semantics.matrix[rule_2_id, :], semantics.matrix[rule_1_id, :])


def compute_rank(model, test_dataset, num_samples=300, filename: Optional[str] = None):
    rule_encoder = test_dataset.rule_encoder

    batch_size = 1
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )

    ranks = [[] for i in range(num_samples)]
    rankless_rules = []

    for bidx, batch in enumerate(tqdm.auto.tqdm(test_dataloader)):

        (tables, table_labels, structures_encoded, labels, rules_encoded) = batch

        rule = rule_encoder.inverse_transform(rules_encoded)[0]

        model_args = {
            "tables": tables,
            "table_labels": table_labels,
            # "structures":structures_encoded
        }

        model_args_batch_idx = {"tables": 1, "table_labels": 1, "structures": 1}

        sampler = BeamSearcher(
            model=model,
            num_beams=num_samples,
            max_rule_length=13,
            model_args_batch_idx=model_args_batch_idx,
            autoregressive_arg="rules",
            rule_encoder=rule_encoder,
        )

        rules_encoded, rule_probabilities = sampler.sample(**model_args)
        predicted_rules = rule_encoder.inverse_transform(
            rules_encoded.view(-1, batch_size * num_samples)
        )

        rankless = True
        for rank, predicted_rule in enumerate(predicted_rules):
            rule_probability = rule_probabilities[0, rank].item()
            if check_equivalence(predicted_rule, rule):
                ranks[rank].append(
                    {
                        "rule": rule,
                        "prediction": predicted_rule,
                        "confidence": rule_probability,
                    }
                )
                rankless = False
                break

        if rankless:
            rankless_rules.append(rule)

    ranks_distribution = [len(ranks[i]) for i in range(len(ranks))]

    result_data = {
        "rules-per-rank": ranks,
        "ranks-distribution": ranks_distribution,
        "rankless-rules": rankless_rules,
    }

    if filename is not None:
        with open(filename, "w") as f:
            json.dump(result_data, f)

    return ranks, ranks_distribution, rankless


import typer
import torch
from zendo import AwareEmpiricistDataset
from zendo import LitAwareEmpiricistModel

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
ZendoSemantics.set_device(DEVICE)

from os import listdir
from os.path import isfile, join, exists


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


import typer


def main(
    training_dir: str = join(root_dir, "training_results"),
    output_dir: str = join(root_dir, "rank_analysis_results"),
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
        "num_samples": 1,
        "rule_filter": lambda x: not ("at_the_left_of" in x),
    }
    test_dataset = AwareEmpiricistDataset(**dataset_args)

    training_conscious = join(training_dir, "empiricist_conscious")

    os.mkdir(output_dir)
    for size in dataset_sizes:
        # load models
        model_conscious = get_model_checkpoint(
            checkpoint_directory=join(training_conscious, f"{size}/checkpoints")
        )
        conjecture_generator = LitAwareEmpiricistModel.load_from_checkpoint(
            model_conscious
        )

        conjecture_generator.to(DEVICE)
        conjecture_generator.eval()

        filename = join(output_dir, f"conscious-{size}.json")
        compute_rank(
            model=conjecture_generator,
            test_dataset=test_dataset,
            num_samples=num_guesses,
            filename=filename,
        )


if __name__ == "__main__":
    typer.run(main)
