import itertools
import random
from sys import getsizeof

from qrennd.configs import Config
from qrennd.layouts import Layout

from .generators import dataset_generator
from .preprocessing import (
    to_model_input,
    to_inputs,
    add_classical_meas_noise,
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
    noise_params = config.dataset.get("classical_meas_noise_anc")

    datasets = list(
        dataset_generator(dataset_dir, experiment_name, basis, **dataset_params)
    )
    proj_matrix = layout.projection_matrix(stab_type)

    # Convert to desired input
    datasets = [add_classical_meas_noise(dataset, noise_params) for dataset in datasets]
    processed = [
        to_inputs(dataset, proj_matrix, input_names=input_names) for dataset in datasets
    ]
    del datasets

    # Process for keras.model input
    inputs = [to_model_input(*arrs, data_type=data_type) for arrs in processed]
    del processed

    return inputs
