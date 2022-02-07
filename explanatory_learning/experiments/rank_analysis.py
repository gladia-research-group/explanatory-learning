import os
import json
from pathlib import Path
from typing import Union, Optional, Tuple, Sequence

import torch
import tqdm
from torch.utils.data import DataLoader

import typer

from explanatory_learning.data.encoders import RuleEncoder
from explanatory_learning.experiments.utils import (
    validate_experiment_input,
    load_model,
    check_equivalence,
)
from explanatory_learning.learning.datasets import AwareEmpiricistDataset
from explanatory_learning.learning.models import LitAwareEmpiricistModel, BeamSearcher
from explanatory_learning.utils import ROOT_DIR


def compute_rules(
    model: LitAwareEmpiricistModel,
    tables: torch.Tensor,
    table_labels: torch.Tensor,
    num_samples: int,
    rule_encoder: RuleEncoder,
) -> Tuple[Sequence[str], Sequence[int]]:

    sampler = BeamSearcher(
        model=model,
        num_beams=num_samples,
        max_rule_length=13,
        model_args_batch_idx={"tables": 1, "table_labels": 1, "structures": 1},
        autoregressive_arg="rules",
        rule_encoder=rule_encoder,
    )

    rules_encoded, rule_probabilities = sampler.sample(
        tables=tables, table_labels=table_labels
    )
    assert rules_encoded.shape[1] == rule_probabilities.shape[0] == 1
    assert rules_encoded.shape[-1] == rule_probabilities.shape[1] == num_samples

    predicted_rules = rule_encoder.inverse_transform(
        rules_encoded.view(-1, num_samples)
    )
    return predicted_rules, rule_probabilities.detach().cpu().numpy()


def compute_rule_rank(predicted_rules: Sequence[str], rule: str) -> Optional[int]:
    for rank, predicted_rule in enumerate(predicted_rules):
        if check_equivalence(predicted_rule, rule):
            return rank
    return None


def compute_ranks(
    model: LitAwareEmpiricistModel,
    test_dataset: AwareEmpiricistDataset,
    num_samples: int = 300,
    filename: Union[Path, str, None] = None,
):
    rule_encoder = test_dataset.rule_encoder

    batch_size = 1
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )

    ranks = [[] for _ in range(num_samples)]
    rankless_rules = []
    for bidx, batch in enumerate(tqdm.auto.tqdm(test_dataloader)):

        (tables, table_labels, _, _, rules_encoded) = batch
        predicted_rules, rule_probabilities = compute_rules(
            model=model,
            tables=tables,
            table_labels=table_labels,
            num_samples=num_samples,
            rule_encoder=rule_encoder,
        )

        rule = rule_encoder.inverse_transform(rules_encoded)[0]
        rank = compute_rule_rank(predicted_rules, rule)

        if rank is not None:
            ranks[rank].append(
                {
                    "rule": rule,
                    "prediction": predicted_rules[rank],
                    "confidence": rule_probabilities[rank],
                }
            )
        else:
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

    return ranks, ranks_distribution, rankless_rules


def main(
    training_dir: Path = ROOT_DIR / "training_results",
    output_dir: Path = ROOT_DIR / "rank_analysis_results",
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
        conjecture_generator = load_model(
            training_dir / f"empiricist_conscious/{size}/checkpoints",
            LitAwareEmpiricistModel,
        )

        compute_ranks(
            model=conjecture_generator,
            test_dataset=test_dataset,
            num_samples=config["num_beams"],
            filename=output_dir / f"conscious-{size}.json",
        )


if __name__ == "__main__":
    typer.run(main)
