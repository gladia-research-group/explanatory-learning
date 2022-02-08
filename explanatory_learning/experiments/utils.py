from os import listdir
from pathlib import Path
from typing import Union

import torch
import typer

from explanatory_learning.data.utils import ZendoSemantics

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
ZendoSemantics.set_device(DEVICE)


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


def validate_experiment_input(training_dir: Path, output_dir: Path, config_file: Path):
    # validate input
    if not training_dir.exists():
        typer.echo(
            message=f"Error: directory {training_dir} does not exists! Run 'training.py' to create!",
            err=True,
        )
        raise typer.Exit()

    if output_dir.exists():
        typer.echo(message=f"Error: directory {output_dir} exists already!", err=True)
        raise typer.Exit()

    if not config_file.exists():
        typer.echo(message=f"Error: file {config_file} doesn't exist!", err=True)
        raise typer.Exit()


def get_checkpoint(
    checkpoint_directory: Union[str, Path], type: str = "complete"
) -> Path:
    checkpoint_directory = Path(checkpoint_directory)
    onlyfiles = sorted(
        [
            file
            for file in listdir(checkpoint_directory)
            if (checkpoint_directory / file).is_file()
        ]
    )

    for file in onlyfiles:
        file_parts = file.split("-")
        if file_parts[0] == type:
            full_filename = checkpoint_directory / file
            return full_filename
    assert False


def load_model(checkpoint_directory: Path, model_class: type) -> torch.nn.Module:
    model = model_class.load_from_checkpoint(str(get_checkpoint(checkpoint_directory)))
    model.to(DEVICE)
    model.eval()
    return model
