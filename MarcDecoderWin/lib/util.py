import itertools
import random

from qrennd.configs import Config
from qrennd.layouts import Layout

from .generators import dataset_generator
from .preprocessing import (
    to_model_input,
    to_inputs,
)
from .sequences import Sequence


def load_datasets(config: Config, layout: Layout, dataset_name: str):
    batch_size = config.train["batch_size"]
    experiment_name = config.dataset["folder_format_name"]

    input_names = config.dataset["input_names"]
    data_type = int if "outcomes" in input_names else bool

    basis = config.dataset["basis"]
    stab_type = f"{basis.lower()}_type"

    dataset_dir = config.experiment_dir / dataset_name
    dataset_params = config.dataset[dataset_name]
    dataset_params["distance"] = config.dataset["distance"]

    datasets = list(
        dataset_generator(dataset_dir, experiment_name, basis, **dataset_params)
    )
    proj_matrix = layout.projection_matrix(stab_type)
    # Convert to desired input
    processed = [
        to_inputs(dataset, proj_matrix, input_names=input_names) for dataset in datasets
    ]

    # Process for keras.model input
    input = [to_model_input(*arrs, data_type=data_type) for arrs in processed]
    #
    # sequences = (Sequence(*tensors, batch_size) for tensors in input)
    # sequences = ((b for b in sequence) for sequence in sequences)
    # sequences_flattened = itertools.chain.from_iterable(sequences)
    # sequences_flattened = list(sequences_flattened)

    return input


def load_datasets_backup(
    config: Config,
    layout: Layout,
    dataset_name: str,
):
    batch_size = config.train["batch_size"]
    experiment_name = config.dataset["folder_format_name"]

    input_names = config.dataset["input_names"]
    data_type = int if "outcomes" in input_names else bool

    basis = config.dataset["basis"]
    stab_type = f"{basis.lower()}_type"

    dataset_dir = config.experiment_dir / dataset_name
    dataset_params = config.dataset[dataset_name]
    dataset_params["distance"] = config.dataset["distance"]

    dataset_gen = dataset_generator(
        dataset_dir, experiment_name, basis, **dataset_params
    )
    proj_matrix = layout.projection_matrix(stab_type)

    # Convert to desired input
    processed_gen = (
        to_inputs(dataset, proj_matrix, input_names=input_names)
        for dataset in dataset_gen
    )

    # Process for keras.model input
    input_gen = (to_model_input(*arrs, data_type=data_type) for arrs in processed_gen)

    sequences = (Sequence(*tensors, batch_size) for tensors in input_gen)
    sequences = ((b for b in sequence) for sequence in sequences)
    sequences_flattened = itertools.chain.from_iterable(sequences)

    def infinite_dataset():
        while True:
            yield from sequences_flattened

    final_sequence = (c for c in infinite_dataset())
    return final_sequence
