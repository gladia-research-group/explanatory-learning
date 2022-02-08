import gc
import json
import math
import os
from collections import OrderedDict
from os.path import join
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import typer
from explanatory_learning.learning.datasets import (
    FalsifierDataset,
    RandomizedAwareDataset,
    RandomizedEmpiricistDataset,
)
from explanatory_learning.learning.models import (
    LitAwareEmpiricistModel,
    LitEmpiricistModel,
    LitEmpiricistRuleModel,
    LitFalsifierModel,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset

from explanatory_learning.utils import ROOT_DIR


MAX_RULE_NUMBER = 1438
MAX_STRUCT_NUMBER = 10000
VAL_STRUCT_NUMBER = 3000
NORMALIZED_EPOCH_NUMBER = 2


def datasets_from_configurations(
    named_configurations: List[Tuple[str, dict]], dataset_class
):
    datasets = OrderedDict()
    for config_name, config in named_configurations:
        assert type(config_name) == str
        assert type(config) == dict
        datasets[config_name] = dataset_class(**config)
        gc.collect()
    return datasets


def train_model(
    model_class,
    model_args: Dict[str, Any],
    output_directory: str,
    monitor_variable: str,
    train_datasets: Dict[str, Dataset],  # <- should be ordereddict
    val_datasets: Dict[str, Dataset],  # <- should be ordereddict
    batch_size: int = 512,
    val_batch_size: int = 1024,
    val_epoch_interval: int = 1,
    num_epochs: int = 5,
    early_stop: bool = True,
):

    config_types = list(train_datasets.keys())

    experiment_dir = output_directory
    checkpoint_dir = join(output_directory, "checkpoints")
    logging_dir = output_directory

    train_info_path = join(experiment_dir, "training_info.json")
    val_info_path = join(experiment_dir, "validation_info.json")

    train_data = {
        config_type: {
            "num_rules": len(train_datasets[config_type].rules),
            "structs_per_rule": train_datasets[config_type].num_samples,
        }
        for config_type in config_types
    }

    val_data = {
        config_type: {
            "num_rules": len(val_datasets[config_type].rules),
            "structs_per_rule": val_datasets[config_type].num_samples,
        }
        for config_type in config_types
    }

    # store in memory
    os.mkdir(experiment_dir)
    with open(train_info_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(val_info_path, "w") as f:
        json.dump(val_data, f, indent=2)

    model = model_class(**model_args)
    torch.manual_seed(0)
    for config_type in config_types:

        print(f"training on configuration: {config_type}")

        # get datasets
        train_dataset = train_datasets[config_type]
        val_dataset = val_datasets[config_type]

        # instantiate loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
        )

        # instantiate callbacks
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=monitor_variable,
            dirpath=checkpoint_dir,
            filename=config_type + "-{epoch}-{" + monitor_variable + ":.5f}",
            save_top_k=1,
            period=val_epoch_interval,
            mode="min",
        )

        callbacks = [checkpoint_callback]
        if early_stop:
            early_stop_callback = EarlyStopping(
                monitor=monitor_variable,
                min_delta=0.00,
                patience=0,
                verbose=False,
                mode="min",
                check_on_train_epoch_end=False,
            )
            early_stop_callback.based_on_eval_results = True
            callbacks.append(early_stop_callback)

        # instantiate loggers
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=logging_dir, version=config_type
        )

        # create trainer
        trainer = pl.Trainer(
            gpus=int(torch.cuda.is_available()),
            min_epochs=1,
            max_epochs=num_epochs,
            progress_bar_refresh_rate=60,
            logger=tb_logger,
            log_every_n_steps=150,
            check_val_every_n_epoch=val_epoch_interval,
            callbacks=callbacks,
        )

        # train
        trainer.fit(model, train_loader, val_loader)

        # load best model ------------------------------------------------------
        model = model_class.load_from_checkpoint(checkpoint_callback.best_model_path)
    return model, trainer


def main(
    output_folder: str = str(ROOT_DIR / "training_results"),
    config_file: Optional[str] = str(ROOT_DIR / "default.json"),
):

    # check that output_folder doesn't exists
    if os.path.exists(output_folder):
        typer.echo(
            message=f"Error: directory {output_folder} exists already!", err=True
        )
        raise typer.Exit()

    with open(config_file, "r") as f:
        config_data = json.load(f)

    datasets_train = config_data["train_datasets"]
    dataset_val = config_data["val_dataset"]
    models_args = config_data["model_args"]
    training_sets = config_data["training_sets"]
    val_structures = 1000

    # load validation dataset
    validation_configurations = [
        (
            "complete",
            {
                "json_file": dataset_val,
                "num_samples": val_structures,
                "rule_filter": lambda x: not ("at_the_left_of" in x),
            },
        )
    ]

    # datasets
    model_types = {
        "empiricist_rules": (
            LitEmpiricistRuleModel,
            RandomizedAwareDataset,
            "val_loss",
        ),
        "empiricist_labels": (
            LitEmpiricistModel,
            RandomizedEmpiricistDataset,
            "val_loss",
        ),
        "empiricist_conscious": (
            LitAwareEmpiricistModel,
            RandomizedAwareDataset,
            "val_loss_structures",
        ),
        "interpreter": (LitFalsifierModel, FalsifierDataset, "val_loss"),
    }

    os.mkdir(output_folder)
    assert models_args.keys() == model_types.keys()  # sanity check
    for model_type in model_types:
        model_class, dataset_class, monitor_variable = model_types[model_type]
        model_args = models_args[model_type]

        # load validation set for model
        val_datasets = datasets_from_configurations(
            validation_configurations, dataset_class
        )

        output_folder_model = os.path.join(output_folder, model_type)
        os.mkdir(output_folder_model)
        for training_set in training_sets:
            num_structures = training_set["structures"]
            num_rules = training_set["rules"]
            output_folder_tset = os.path.join(
                output_folder_model, f"{num_structures}structures-{num_rules}rules"
            )

            dataset_train = datasets_train[str(num_rules)]
            train_configurations = [
                (
                    "complete",
                    {"json_file": dataset_train, "num_samples": num_structures},
                )
            ]

            train_datasets = datasets_from_configurations(
                train_configurations, dataset_class
            )

            # normalize number of epochs and validation interval
            assert num_structures * num_rules <= MAX_STRUCT_NUMBER * MAX_RULE_NUMBER
            epoch_number = int(
                math.ceil(
                    NORMALIZED_EPOCH_NUMBER
                    * (MAX_STRUCT_NUMBER / num_structures)
                    * (MAX_RULE_NUMBER / num_rules)
                )
            )
            val_epoch_interval = int(
                math.ceil(
                    VAL_STRUCT_NUMBER / num_structures * MAX_RULE_NUMBER / num_rules
                )
            )

            train_model(
                model_class=model_class,
                model_args=model_args,
                output_directory=output_folder_tset,
                monitor_variable=monitor_variable,
                train_datasets=train_datasets,
                val_datasets=val_datasets,
                batch_size=512,
                val_batch_size=1024,
                val_epoch_interval=val_epoch_interval,
                num_epochs=epoch_number,
                early_stop=True,
            )


if __name__ == "__main__":
    typer.run(main)
