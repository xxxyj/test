import itertools
import numpy as np

from qrennd.configs import Config
from qrennd.layouts import Layout

from .generators import dataset_generator
from .preprocessing import (
    to_model_input,
    to_inputs,
)
from .sequences import Sequence


def _pad_and_concat(arrays: tuple[np.ndarray, ...]) -> np.ndarray:
    """Pad arrays along non-batch dimensions before concatenation.

    Each array is assumed to have the same rank and its first dimension is the
    batch dimension. The remaining dimensions are padded with zeros so that all
    arrays share a common shape, allowing them to be concatenated along the
    batch axis without shape mismatches.
    """

    if not arrays:
        return np.array([])

    ndims = arrays[0].ndim
    # Compute the target size for every non-batch dimension
    max_shape = [max(arr.shape[i] for arr in arrays) for i in range(1, ndims)]

    padded = []
    for arr in arrays:
        pad_width = [(0, 0)]
        pad_width += [
            (0, max_dim - arr.shape[i + 1]) for i, max_dim in enumerate(max_shape)
        ]
        padded.append(np.pad(arr, pad_width, mode="constant"))

    return np.concatenate(padded, axis=0)


def load_datasets(config: Config, layout: Layout, dataset_name: str):
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

    # Process for keras.model input and concatenate for tf.data
    model_inputs = [to_model_input(*arrs, data_type=data_type) for arrs in processed]
    rec_inputs, eval_inputs, log_errors = zip(*model_inputs)
    rec_inputs = _pad_and_concat(rec_inputs)
    eval_inputs = _pad_and_concat(eval_inputs)
    log_errors = np.concatenate(log_errors, axis=0)

    return rec_inputs, eval_inputs, log_errors


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
