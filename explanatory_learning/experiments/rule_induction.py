import json
import os
from os.path import join
from pathlib import Path
from typing import Union, Optional, List

import torch
import tqdm

import typer

from explanatory_learning.data.encoders import RuleEncoder
from explanatory_learning.experiments.utils import (
    validate_experiment_input,
    check_equivalence,
    DEVICE,
    load_model,
)
from explanatory_learning.learning.datasets import AwareEmpiricistDataset
from explanatory_learning.learning.models.transformers import (
    LitAwareEmpiricistModel,
    LitEmpiricistRuleModel,
    LitFalsifierModel,
    RationalistModel,
)
from explanatory_learning.learning.samplers import BeamSearcher
from explanatory_learning.utils import ROOT_DIR


aware_batch2dict = lambda x: {
    "tables": x[0],
    "table_labels": x[1],
    "structures": x[2],
    "labels": x[3],
    "rules": x[4],
}


def get_rationalist_rule(model: RationalistModel, rule_encoder: RuleEncoder) -> str:
    pred_rules = rule_encoder.inverse_transform(model.hidden_rule)
    assert len(pred_rules) == 1
    return pred_rules[0]


def get_aware_rule(model: RationalistModel, rule_encoder: RuleEncoder) -> str:
    pred_rules = rule_encoder.inverse_transform(model.most_probable_rule)
    assert len(pred_rules) == 1
    return pred_rules[0]


def get_empiricist_rule(
    beam_searcher: BeamSearcher,
    rule_encoder: RuleEncoder,
    tables,
    table_labels,
    num_samples: int,
) -> str:
    rules_encoded, _ = beam_searcher.sample(tables=tables, table_labels=table_labels)
    most_probable_rule = rules_encoded.view(13, num_samples)[:, 0:1]
    pred_rules = rule_encoder.inverse_transform(most_probable_rule)
    return pred_rules[0]


def get_batch(rule: str, dataset: AwareEmpiricistDataset) -> List[torch.Tensor]:
    rule_id = dataset.id_from_rule(rule)
    idx = dataset.rule_id_to_idx(rule_id)
    batch = dataset.collate_fn([dataset[idx]])
    batch = [x.to(DEVICE) for x in batch]
    return batch


def metrics(
    num_guesses: int,
    falsifier: LitFalsifierModel,
    conjecture_generator: LitAwareEmpiricistModel,
    empiricist: LitEmpiricistRuleModel,
    dataset: AwareEmpiricistDataset,
    filename: Optional[str] = None,
):
    rule_encoder = dataset.rule_encoder
    rationalist = RationalistModel(
        num_beams=num_guesses,
        falsifier=falsifier,
        conjectures_generator=conjecture_generator,
        max_rule_length=dataset._encoded_rules.shape[-1],
        rule_encoder=rule_encoder,
    )

    bsearch = BeamSearcher(
        model=empiricist,
        num_beams=num_guesses,
        max_rule_length=13,
        model_args_batch_idx={"tables": 1, "table_labels": 1, "structures": 1},
        autoregressive_arg="rules",
        rule_encoder=rule_encoder,
    )

    sum_accuracy_aware = 0
    sum_accuracy_empiricist = 0
    sum_accuracy_rationalist = 0
    for i, rule in enumerate(tqdm.auto.tqdm(dataset.rules)):
        (
            tables,
            table_labels,
            structures_encoded,
            labels,
            rules_encoded,
        ) = get_batch(rule, dataset)

        # guess rule in rationalist model
        rationalist.guess_rule(tables, table_labels, rule, find_nearest=True)

        # get predictions
        pred_rationalist = get_rationalist_rule(rationalist, rule_encoder)
        pred_aware = get_aware_rule(rationalist, rule_encoder)
        pred_empiricist = get_empiricist_rule(
            bsearch, rule_encoder, tables, table_labels, num_guesses
        )

        sum_accuracy_aware += check_equivalence(pred_aware, rule)
        sum_accuracy_empiricist += check_equivalence(pred_empiricist, rule)
        sum_accuracy_rationalist += check_equivalence(pred_rationalist, rule)

    num_rules = len(dataset.rules)
    accuracy_aware = sum_accuracy_aware / num_rules
    accuracy_empiricist = sum_accuracy_empiricist / num_rules
    accuracy_rationalist = sum_accuracy_rationalist / num_rules

    result_data = {
        "conscious-accuracy": accuracy_aware,
        "empiricist-accuracy": accuracy_empiricist,
        "rationalist-accuracy": accuracy_rationalist,
    }

    if filename is not None:
        with open(filename, "w") as f:
            json.dump(result_data, f)

    return result_data


def main(
    training_dir: Path = ROOT_DIR / "training_results",
    output_dir: Path = ROOT_DIR / "rule_induction_results",
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
            "num_samples": 1,
            "rule_filter": lambda x: not ("at_the_left_of" in x),
        }
    )

    os.mkdir(output_dir)
    for size in dataset_sizes:
        # load models
        conjecture_generator = load_model(
            training_dir / f"empiricist_conscious/{size}/checkpoints",
            model_class=LitAwareEmpiricistModel,
        )

        interpreter = load_model(
            training_dir / f"interpreter/{size}/checkpoints",
            model_class=LitFalsifierModel,
        )

        empiricist = load_model(
            training_dir / f"empiricist_rules/{size}/checkpoints",
            model_class=LitEmpiricistRuleModel,
        )

        # compute metrics
        metrics(
            num_guesses=config["num_beams"],
            empiricist=empiricist,
            falsifier=interpreter,
            conjecture_generator=conjecture_generator,
            dataset=test_dataset,
            filename=join(output_dir, f"{size}.json"),
        )


if __name__ == "__main__":
    typer.run(main)
